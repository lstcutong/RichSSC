import os
import argparse
import numpy as np
from tqdm import tqdm
import sscMetrics
import warnings
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import center_sample_points, extent_points_per_voxel
from interpolate import *
import glob
from config import Path
from visualize import convert_voxel_into_open3dmesh, colorMap
import open3d as o3d

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch SSC Training')
parser.add_argument('--dataset', type=str, default='nyucad', choices=['nyu', 'nyucad', 'debug', 'suncg'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='ddr',
                    choices=['aic', 'ddr'],
                    help='encoder name (default: aic)')
parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--fc_nblocks', type=int, default=3, help='layer num of point decoder')
parser.add_argument('--pretrain_en', metavar='DIR', help='pretrained encoder')
parser.add_argument('--use_cls', default=True, type=bool)
parser.add_argument('--scale', default=4, type=int, help="for downscale, base res:(240,144,240)")
parser.add_argument('--sample_mode', default="vote", type=str)
parser.add_argument('--point_to_extent', default=15, type=int)
parser.add_argument('--visualize', action="store_true")
parser.add_argument('--base_folder', type=str, default="", help="basic folder for saving visualizing results")

global args
args = parser.parse_args()


class NYUDataset_Eval(torch.utils.data.Dataset):
    def __init__(self, root, scale_factor=4, sample_mode="center", point_to_extent=15):
        self.param = {'voxel_size': (240, 144, 240),
                      'voxel_unit': 0.02,  # 0.02m, length of each grid == 20mm
                      'cam_k': [[518.8579, 0, 320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
                                [0, 518.8579, 240],  # cx = K(1,3); cy = K(2,3);
                                [0, 0, 1]],  # fx = K(1,1); fy = K(2,2);
                      }
        #
        self.subfix = 'npz'
        self.filepaths = self.get_filelist(root, self.subfix)
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] \
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        self.transforms_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.scale_factor = scale_factor  # for downsample, not support for upsample
        self.sample_mode = sample_mode
        self.point_to_extent = point_to_extent

        print('Dataset:{} files'.format(len(self.filepaths)))

    def get_item_in_running_sample(self, index):
        filepath = self.filepaths[index]
        filename = os.path.split(filepath)[1]
        _name = filename[:-4]
        npz_file = np.load(self.filepaths[index])
        rgb_tensor = npz_file['rgb']
        depth_tensor = npz_file['depth']
        tsdf_tensor = npz_file['tsdf_hr']
        position = npz_file['position']
        target_hr = npz_file['target_hr']

        if self.scale_factor == 1:
            gt_tsdf, gt_label = tsdf_tensor, target_hr
        else:
            gt_tsdf, gt_label = downsample_tsdf(tsdf_tensor, self.scale_factor), downsample_vox_labelv2(target_hr,
                                                                                                      downscale=self.scale_factor)

        test_points, test_label = center_sample_points(gt_label, self.scale_factor * self.param["voxel_unit"])
        if self.sample_mode != "center":
            ww, hh, dd = gt_label.shape
            test_points = extent_points_per_voxel(test_points, ww, hh, dd, self.point_to_extent)

        try:
            nonempty = self.get_nonempty2(gt_tsdf, gt_label, 'TSDF')  # 这个更符合SUNCG的做法
        except:
            nonempty = np.ones(gt_label.shape, dtype=np.float32)
            nonempty[gt_tsdf == 255] = 0

        return rgb_tensor, depth_tensor, tsdf_tensor, test_points, test_label, gt_label.T, nonempty.T, position, _name  # 和采点时使用的yxz坐标系对齐

    def __getitem__(self, index):
        return self.get_item_in_running_sample(index)

    def __len__(self):
        return len(self.filepaths)

    def get_filelist(self, root, subfix):
        if root is None:
            raise Exception("Oops! 'root' is None, please set the right file path.")
        _filepaths = list()
        if isinstance(root, list):  # 将多个root
            for i, root_i in enumerate(root):
                fp = glob.glob(root_i + '/*.' + subfix)
                fp.sort()
                _filepaths.extend(fp)
        elif isinstance(root, str):
            _filepaths = glob.glob(root + '/*.' + subfix)  # List all files in data folder
            _filepaths.sort()
        if len(_filepaths) == 0:
            raise Exception("Oops!  That was no valid data in '{}'.".format(root))

        return _filepaths

    @staticmethod
    def get_nonempty2(voxels, target, encoding):  # Get none empty from depth voxels
        data = np.ones(voxels.shape, dtype=np.float32)  # init 1 for none empty
        data[target == 255] = 0
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels == 0] = 0
        elif encoding == 'TSDF':
            data[voxels >= np.float32(0.001)] = 0
            data[voxels == 1] = 1

        return data


def save_vis(pred, save_folder, filename, nonempty):
    """

    Args:
        pred:  [b, W, H, D]
        save_folder:
        filename:
        nonempty: [b, W, H, D]

    Returns:

    """
    for i in range(len(pred)):
        y = pred[i]
        y[np.where(nonempty[i] == 0)] = 0
        yc = colorMap[y]
        mesh = convert_voxel_into_open3dmesh(yc, size=0.8)
        o3d.io.write_triangle_mesh(os.path.join(save_folder, filename[i] + ".obj"), mesh)


def validate_on_test_dataset(model, data_loader, imp_or_vox="imp", save_folder=None):
    """
    Evaluate on validation set.
        model: network with parameters loaded
        date_loader: TEST mode
    """
    model.eval()  # switch to evaluate mode.
    val_acc, val_p, val_r, val_iou, test_losses = 0.0, 0.0, 0.0, 0.0, 0.0
    _C = 12
    val_cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    val_iou_ssc = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    count = 0

    with torch.no_grad():
        # ---- STSDF  depth, input, target, position, _
        for step, (rgb, depth, tsdf, imp_inputs, imp_label, target_vox, nonempty, position, filename) in tqdm(
                enumerate(data_loader), desc='Validating', unit='frame'):
            var_x_depth = depth.float().cuda()
            position = position.long().cuda()
            imp_inputs = imp_inputs.float().cuda()
            var_x_rgb = rgb.float().cuda()
            var_x_tsdf = tsdf.float().cuda()

            if imp_or_vox == "imp":
                y_pred, y_pred_en = model(x_depth=var_x_depth, x_rgb=var_x_rgb, p=position,
                                          query_points=imp_inputs, return_en_res=True, isTest=True,
                                          sample_mode=args.sample_mode)

            else:
                y_pred, y_pred_en = model(x_depth=var_x_depth, x_rgb=var_x_rgb, p=position,
                                          query_points=imp_inputs[:, 0:10, :], return_en_res=True, isTest=True,
                                          sample_mode=args.sample_mode)

            if imp_or_vox == 'imp':
                b, D, H, W = nonempty.shape
                y_pred = y_pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy
                y_true = imp_label.cpu().data.numpy()  # torch tensor to numpy
                b, c, _ = y_pred.shape
                # print(nonempty.shape)
                nonempty = nonempty.numpy().transpose((0, 2, 3, 1)).reshape((b, -1))  # b, D, H, W -> b, H, W, D

                if save_folder is not None and args.visualize:
                    save_vis(np.argmax(y_pred, axis=2).reshape((b, H, W, D)).transpose((0, 2, 1, 3)), save_folder,
                             filename, nonempty.reshape((b, H, W, D)).transpose((0, 2, 1, 3)))

                p, r, iou, acc, iou_sum, cnt_class = validate_on_batch(y_pred, y_true, nonempty)
            else:
                y_pred_en = rescale_vox_score(y_pred_en, 4 / args.scale, mode="nearest")
                y_pred_en = y_pred_en.cpu().data.numpy()  # CUDA to CPU, Variable to numpy
                y_true = target_vox.numpy()  # torch tensor to numpy

                if save_folder is not None and args.visualize:
                    save_vis(np.argmax(y_pred_en, axis=1).transpose((0, 3, 2, 1)), save_folder, filename,
                             nonempty.numpy().transpose((0, 3, 2, 1)))

                b, c, _, _, _ = y_pred_en.shape
                y_pred_en = y_pred_en.reshape((b, c, -1)).transpose((0, 2, 1))
                y_true = y_true.reshape((b, -1))
                nonempty = nonempty.numpy().reshape((b, -1))
                p, r, iou, acc, iou_sum, cnt_class = validate_on_batch(y_pred_en, y_true, nonempty)

            count += 1
            val_acc += acc
            val_p += p
            val_r += r
            val_iou += iou
            val_iou_ssc = np.add(val_iou_ssc, iou_sum)
            val_cnt_class = np.add(val_cnt_class, cnt_class)
            # print('acc_w, acc, p, r, iou', acc_w, acc, p, r, iou)

    val_acc = val_acc / count
    val_p = val_p / count
    val_r = val_r / count
    val_iou = val_iou / count
    val_iou_ssc, val_iou_ssc_mean = sscMetrics.get_iou(val_iou_ssc, val_cnt_class)
    return val_p, val_r, val_iou, val_acc, val_iou_ssc, val_iou_ssc_mean, test_losses / count


def validate_on_batch(predict, target, nonempty=None):  # CPU
    """
        predict: (bs, channels, D, H, W)
        target:  (bs, channels, D, H, W)
    """
    # TODO: validation will increase the usage of GPU memory!!!
    y_pred = predict
    y_true = target
    p, r, iou = sscMetrics.get_score_completion(y_pred, y_true, nonempty)
    # acc, iou_sum, cnt_class = sscMetrics.get_score_semantic_and_completion(y_pred, y_true, stsdf)
    acc, iou_sum, cnt_class, tp_sum, fp_sum, fn_sum = sscMetrics.get_score_semantic_and_completion(y_pred, y_true,
                                                                                                   nonempty)
    # iou = np.divide(iou_sum, cnt_class)
    return p, r, iou, acc, iou_sum, cnt_class


def eval(imp_or_vox="imp"):
    from models.RichSSC import RichSSC
    base_dirs = Path.db_root_dir(args.dataset)

    print('eval data:{}'.format(base_dirs['val']))
    eval_loader = DataLoader(
        dataset=NYUDataset_Eval(base_dirs['val'], scale_factor=args.scale, sample_mode=args.sample_mode),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    net = RichSSC(mlp_layers=args.fc_nblocks, basic_module=args.model, use_cls=args.use_cls).cuda()
    # args.resume="./pretrain_model/aicnet_imp/cpBest_SSC_IMP.pth.tar"
    # print(net)
    net = torch.nn.DataParallel(net)  # Multi-GPU
    # ---- optionally resume from a checkpoint --------- ---------- ----------#
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            cp_states = torch.load(args.resume)
            net.load_state_dict(cp_states['state_dict'], strict=True)

    base_folder = args.base_folder
    if args.scale == 8:
        reso = "30-16-30"
    elif args.scale == 4:
        reso = "60-32-60"
    elif args.scale == 2:
        reso = "120-64-120"
    else:
        reso = "{}_{}-{}-{}".format(args.scale, int(240 / args.scale), int(144 / args.scale), int(240/args.scale))
    save_folder = os.path.join(base_folder, reso, "imp_{}_{}".format(args.sample_mode, args.point_to_extent if args.sample_mode=="vote" else "1") if imp_or_vox == "imp" else "vox")
    if not os.path.exists(save_folder) and args.visualize:
        os.makedirs(save_folder)

    v_prec, v_recall, v_iou, v_acc, v_ssc_iou, v_mean_iou, v_test_loss = validate_on_test_dataset(net, eval_loader,
                                                                                                  imp_or_vox=imp_or_vox, save_folder=save_folder if args.visualize else None)
    print('Validate with TSDF, p {:.1f}, r {:.1f}, IoU {:.1f}'.format(v_prec * 100.0,
                                                                      v_recall * 100.0, v_iou * 100.0))

    print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(v_acc * 100.0, v_mean_iou * 100.0,
                                                                 v_ssc_iou * 100.0))


if __name__ == '__main__':
    print(args)
    eval("imp") # evaluate our method
    eval('vox') # evaluate baseline
