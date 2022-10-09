from .AICNet import SSC_RGBD_AICNet
import torch.nn as nn
import torch
from .MLP import LocalDecoder
from collections import OrderedDict
from .DDRNet import SSC_RGBD_DDRNet
import os
import torch.nn.functional as F
import numpy as np
class RichSSC(nn.Module):
    def __init__(self, cls=12, basic_module="aic", mlp_layers=5, pretrain_encode=None, use_cls=True, args=None):
        super(RichSSC, self).__init__()
        if basic_module == "aic":
            self.encoding = SSC_RGBD_AICNet()
            out_feature = 256
        elif basic_module == "ddr":
            self.encoding = SSC_RGBD_DDRNet()
            out_feature = 320
        else:
            raise NotImplementedError
        self.use_cls = use_cls
        if use_cls:
            out_feature = out_feature + cls

        self.decoding = LocalDecoder(cls=cls, c_dim=out_feature, n_blocks=mlp_layers)

        self.pretrain_encode = pretrain_encode
        if pretrain_encode is not None:
            self.load_pretrain(args)

    def load_pretrain(self, args=None):
        if os.path.exists(self.pretrain_encode):
            param = torch.load(self.pretrain_encode)
            new_state_dict = OrderedDict()
            for k, v in param["state_dict"].items():                
                if k[:7] == "module.":
                    if args.dataset == "nyu": # uncareful to accidently pacakged DDRNet and AICNet in "embedding" when training on NYU dataset
                        name = k[17: ]
                    else:
                        name = k[7:]
                    new_state_dict[name] = v
                else:
                    if args.dataset == "nyu": # uncareful to accidently pacakged DDRNet and AICNet in "embedding" when training on NYU dataset
                        k = k[10: ]
                    new_state_dict[k] = v
            self.encoding.load_state_dict(new_state_dict, strict=True)

    def fx(self, p, c):
        p_nor = p[:, :, None, None].float()

        cf = F.grid_sample(c, p_nor, padding_mode='border', align_corners=True, mode="bilinear").squeeze(
            -1).squeeze(-1)
        return cf.transpose(1, 2)


    def decoding_split_mode(self, test_points, c_feature, batch_test=102400):
        num_points = test_points.shape[1]
        iters = int(num_points / batch_test)
        pred = []
        with torch.no_grad():
            for i in range(iters):
                points = test_points[:, i * batch_test:(i + 1) * batch_test, :]
                out = self.decoding(points, self.fx(points, c_feature))
                pred.append(out)

            if iters * batch_test < num_points:
                points = test_points[:, iters * batch_test:, :]
                out = self.decoding(points, self.fx(points, c_feature))
                pred.append(out)
        return torch.cat(pred, dim=1)

    def forward_test(self, query_points, c_feature, sample_mode="center"):
        if sample_mode == 'center':
            b, pn, c = query_points.shape
            if pn < 102400:
                return self.decoding(query_points, self.fx(query_points, c_feature))
            else:
                return self.decoding_split_mode(query_points, c_feature)
        elif sample_mode == "vote":
            b, pn, en, c = query_points.shape
            query_points = query_points.reshape((b, -1, c))
            if pn * en <= 102400:
                query_condition = self.fx(query_points, c_feature)
                out = self.decoding(query_points, query_condition)
            else:
                out = self.decoding_split_mode(query_points, c_feature)

            out = out.cpu().detach().numpy()
            prediction = out.reshape((b, pn, en, -1))
            prediction = np.argmax(prediction, axis=3).reshape((b * pn, en))
            new_pred = []
            for i in range(len(prediction)):
                '''
                label = np.argmax(np.bincount(prediction[i]))
                one_hot = np.zeros(12)
                one_hot[label] = 1
                new_pred.append(one_hot)
                
                    isFloor = np.count_nonzero(np.where(pred==2)) > 0
                    isCeil = np.count_nonzero(np.where(pred==1)) > 0
                    if isFloor:
                        one_hot[2] = 1
                    elif isCeil:
                        one_hot[1] = 1
                    else:
                '''

                pred = prediction[i] # [en] stores each point's class label

                empty_space_point_num = en - np.count_nonzero(pred)
                if empty_space_point_num > en * 0.6:
                    one_hot = np.zeros(12)
                    one_hot[0] = 1
                else:
                    one_hot = np.zeros(12)
                    label = np.argmax(np.bincount(pred[np.where(pred!=0)]))
                    one_hot[label] = 1
                new_pred.append(one_hot)
            new_pred = np.array(new_pred)
            new_pred = torch.from_numpy(new_pred).reshape((b, pn, -1)).cuda()
            return new_pred

            #out = out.reshape((b, pn, en, -1))
            #out = out.sum(dim=2)


        elif sample_mode == "avg":
            b, pn, en, c = query_points.shape
            query_points = query_points.reshape((b, -1, c))
            query_condition = self.fx(query_points, c_feature)
            query_condition = query_condition.reshape((b, pn, en, -1))
            query_condition = query_condition.mean(dim=2)

            query_points = query_points.reshape((b, pn, en, -1))
            query_points = query_points.mean(dim=2)
            try:
                out = self.decoding(query_points, query_condition)
            except:
                out = self.decoding_split_mode(query_points, query_condition)
            return out

    def forward(self, x_depth=None, x_rgb=None, p=None, query_points=None, return_en_res=False, isTest=False, sample_mode="center"):
        feature, pre_cls_score = self.encoding(x_depth=x_depth, x_rgb=x_rgb, p=p)
        if self.use_cls:
            c_feature = torch.cat([feature, pre_cls_score], dim=1)
        else:
            c_feature = feature

        if isTest:
            out = self.forward_test(query_points, c_feature, sample_mode=sample_mode)
        else:
            out = self.decoding(query_points, self.fx(query_points, c_feature))
        if return_en_res:
            return out, pre_cls_score
        else:
            return out

    def get_trainable_parameters(self, lr_decode=1e-4, finetune_encode=False, lr_encode=1e-4):
        params = [
            {"params": self.decoding.parameters(), "lr": lr_decode}
        ]
        if finetune_encode:
            params.append({
                "params": self.encoding.parameters(), "lr": lr_encode
            })
        else:
            for k, v in self.encoding.named_parameters():
                v.requires_grad = False

        return params
