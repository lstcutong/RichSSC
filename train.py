from turtle import pos
from models.ReS import ReS_SSC
from utils import seed_torch
import os

import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable

import datetime
import matplotlib.pyplot as plt
from dataloaders import make_data_loader, make_dataloader_for_vis
import sscMetrics
from config import colorMap, cls_name

from visualize import *
import config
from models.ReS import ReS_SSC
import sys
import warnings
from losses import *
import yaml

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch SSC Training')
parser.add_argument('--dataset', type=str, default='nyucad', choices=['nyu', 'nyucad', 'debug', 'suncg'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='aic',
                    choices=['aic', "ddr"],
                    help='encoder name (default: aic)')
# parser.add_argument('--data_augment', default=False, type=bool,  help='data augment for training')
# parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default='./checkpoints/debug', metavar='DIR', help='path to checkpoint')
parser.add_argument('--pretrain_en', metavar='DIR', help='path to pretrained encoder')
parser.add_argument('--batch_pointnum', type=int, default=10240, help='number of sample points when training')
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--task', default="ssc", help='scene completion or semantic scene completion')
parser.add_argument('--use_target', default="hr", choices=["lr", "hr"])
parser.add_argument('--fc_nblocks', type=int, default=3)

parser.add_argument('--model_name', default='Res_SSC', type=str, help='name of model to save check points')
parser.add_argument('--vis_every', default=20, type=int, help='visualize results every n epoch')
parser.add_argument('--use_cls', action="store_true")



global args
args = parser.parse_args()
seed_torch(3055)
if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
with open(os.path.join(args.checkpoint, "config.yaml"), 'w') as f:
        yaml.safe_dump(vars(args), f, encoding='utf-8', allow_unicode=True)

def save_logs(train_loss, test_loss, pixel_acc, sc_metrix, mIoU, sscIoU):
    epochs = len(train_loss)
    X = np.arange(1, epochs + 1)

    plt.title("losses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(X, np.array(train_loss), color='red', label="train")
    plt.plot(X, np.array(test_loss), color='blue', label="test")
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint, "losses.png"))
    plt.clf()

    plt.title("acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.plot(X, np.array(pixel_acc), color='red', label="pixel-acc")
    plt.plot(X, np.array(mIoU), color="green", label="mIoU")
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint, "acc.png"))
    plt.clf()

    plt.title("sc_metrix")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.plot(X, np.array(sc_metrix[0]), color='red', label="precision")
    plt.plot(X, np.array(sc_metrix[1]), color="green", label="recall")
    plt.plot(X, np.array(sc_metrix[2]), color="blue", label="IoU")
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint, "sc_metrix.png"))
    plt.clf()

    plt.title("ssc_metrix")
    plt.xlabel("epoch")
    plt.ylabel("iou")
    for i in range(len(sscIoU)):
        plt.plot(X, np.array(sscIoU[i]), color=np.array(colorMap[i]) / 255, label=cls_name[i])
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint, "ssc_metrix.png"))
    plt.clf()

    with open(os.path.join(args.checkpoint, "logs.txt"), "w") as f:
        f.write("the best mIoU is {}".format(np.max(np.array(mIoU))))
    
    


train_loader_vis, val_loader_vis = make_dataloader_for_vis(args)
loaders_vis = [("train", train_loader_vis),
           ("val", val_loader_vis)]

def vis_res(model, epoch):
    model.eval()
    W, H, D = 60, 36, 60
    global loaders_vis

    with torch.no_grad():
        for desc, loader in loaders_vis:
            vis_folder = os.path.join(args.checkpoint, "epoch_{}".format(epoch), desc)
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
        # ---- STSDF  depth, input, target, position, _
            for step, (rgb, depth, tsdf, test_points, y_true, target_lr, nonempty, position, filename) in tqdm(
                    enumerate(loader), desc='Visualizing {}'.format(desc), unit='frame'):
                var_x_depth = depth.float().cuda()
                position = position.long().cuda()
                test_points = test_points.float().cuda()
                var_x_rgb = rgb.float().cuda()
                var_x_tsdf = tsdf.float().cuda()

                if args.model in ["RGBTSDFUnet", "light", "TSDFUNet"]:
                    y_pred = model(x_tsdf=var_x_tsdf, x_rgb=var_x_rgb, p=position, query_points=test_points)
                else:
                    y_pred = model(x_depth=var_x_depth, x_rgb=var_x_rgb, p=position, query_points=test_points)

                y_pred = y_pred.detach().cpu().numpy()[0]
                y_pred = np.argmax(y_pred, axis=1)

                y_pred = y_pred.reshape((H,W,D)).transpose((1, 0, 2))
                freespace = np.where(y_true.numpy()[0].reshape((H,W,D)).transpose((1, 0, 2)) == 255)
                y_pred[freespace] = 0

                voxel = colorMap[y_pred.astype(np.int)]
                try:
                    mesh = convert_voxel_into_open3dmesh(voxel, size=0.8)

                #print(os.path.join(vis_folder, filename[0] + ".obj"))
                    o3d.io.write_triangle_mesh(os.path.join(vis_folder, filename[0] + ".obj"), mesh)
                except:
                    o3d.io.write_triangle_mesh(os.path.join(vis_folder, filename[0] + ".obj"), o3d.geometry.TriangleMesh.create_sphere())



def train():
    # ---- create model ---------- ---------- ---------- ---------- ----------#
    print(args)
    net = ReS_SSC(basic_module=args.model, pretrain_encode=args.pretrain_en, mlp_layers=args.fc_nblocks, use_cls=args.use_cls).cuda()
    trainable_param = net.get_trainable_parameters(lr_decode=args.lr, finetune_encode=False, lr_encode=1e-5)
    print(net)
    net = torch.nn.DataParallel(net)  # Multi-GPU
    smooth = 0.0
    # ---- optionally resume from a checkpoint --------- ---------- ----------#
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            cp_states = torch.load(args.resume)
            net.load_state_dict(cp_states['state_dict'], strict=True)
        else:
            raise Exception("=> NO checkpoint found at '{}'".format(args.resume))

    # -------- ---------- --- Set checkpoint --------- ---------- ----------#
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H.%M.%S")
    # model_info = 'epoch{}_lr{}'.format(args.epochs, args.lr)
    
    cp_filename = os.path.join(args.checkpoint, 'cp_{}.pth.tar'.format(args.model_name))
    cp_best_filename = os.path.join(args.checkpoint, 'cpBest_{}.pth.tar'.format(args.model_name))

    # ---- Define loss function (criterion) and optimizer ---------- ----------#

    if args.resume:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
    else:
        optimizer = torch.optim.SGD(trainable_param, lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)


    semantic_loss = torch.nn.CrossEntropyLoss(weight=config.class_weights, ignore_index=255, label_smoothing=smooth).cuda()

    # ---- Print Settings for training -------- ---------- ---------- ----------#
    print('\nInitial Learning rate:{} \nBatch size:{} \nNumber of workers:{}'.format(
        args.lr,
        args.batch_size,
        args.workers,
        cp_filename))
    print("Checkpoint filename:{}".format(cp_filename))

    # ---- Data loader
    train_loader, val_loader = make_data_loader(args)

    np.set_printoptions(precision=1)

    # ---- Train
    step_count_all = 0
    best_miou = 0

    print("Start training")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4,
                                                           threshold=1e-4)
    stop = False
    epoch = 0

    _train_loss, _test_loss, _pixel_acc, _sc_metrix, _mIoU, _sscIoU = [], [], [], [], [], []
    _C = 12 if args.task == "ssc" else 2

    while not stop:
        if optimizer.param_groups[0]['lr'] < 1e-4:
            stop = True
        net.train()  # switch to train mode
        decs_str = 'Training epoch {}, current lr {}'.format(epoch + 1, optimizer.param_groups[0]['lr'])
        log_loss_1epoch = 0.0
        step_count = 0

        torch.cuda.empty_cache()
        loss_to_show = 0.0
        loss_geo_to_show = 0.0
        loss_ssc_to_show = 0.0
        for step, (rgb, depth, tsdf, train_points, train_label, position, _) in enumerate(train_loader):
            y_true = train_label.long().contiguous()
            y_true = y_true.view(-1).cuda()  # bs * point_num

            # ---- (bs, C, D, H, W), channel first for Conv3d in pyTorch
            # FloatTensor to Variable. (bs, channels, 240L, 144L, 240L)
            x_depth = depth.float().cuda()
            position = position.long().cuda()
            train_points = train_points.float().cuda()
            x_rgb = rgb.float().cuda()
            x_tsdf = tsdf.float().cuda()


            y_pred = net(x_depth=x_depth, x_rgb=x_rgb, p=position, query_points=train_points)

            y_pred = y_pred.contiguous()
            y_pred = y_pred.view(-1, _C)  # C = 12
            #print("{} {}".format(epoch, y_pred))
            # print(y_pred[20000], y_true[20000])

            optimizer.zero_grad()
            if args.task == "ssc":
                loss_ssc = semantic_loss(y_pred, y_true)
                loss_geo = geometry_loss(y_pred, y_true, smooth=smooth)
                loss = loss_ssc + 2 * loss_geo
                #loss = loss_ssc
            elif args.task == "sc":
                loss_geo = geometry_loss(y_pred, y_true, smooth=smooth)
                loss = loss_geo

            loss.backward()
            optimizer.step()
            #print("grad",net.module.decoding.linear.weight.grad)
            loss_to_show += loss.item()
            if args.task == "ssc":
                loss_ssc_to_show += loss_ssc.item()
            loss_geo_to_show += loss_geo.item()

            if args.task == "ssc":
                sys.stdout.write(
                    "{} loss {:.6f} loss_ssc:{:.6f} loss_geo:{:.6f} step {}\r".format(decs_str, loss, loss_ssc,
                                                                                      loss_geo, step + 1))
            else:
                sys.stdout.write("{} loss {:.6f} step {}\r".format(decs_str, loss, step + 1))
            sys.stdout.flush()

        print("")
        if args.task == "ssc":
            print("training epoch {} loss {:.6f} loss_ssc {:.6f} loss_geo {:.6f}".format(epoch + 1,
                                                                                         loss_to_show / (step + 1),
                                                                                         loss_ssc_to_show / (step + 1),
                                                                                         loss_geo_to_show / (step + 1)))
        else:
            print("training epoch {} loss {:.6f}".format(epoch + 1, loss_to_show / (step + 1)))
        # ---- Evaluate on validation set
        v_prec, v_recall, v_iou, v_acc, v_ssc_iou, v_mean_iou, v_test_loss = validate_on_test_dataset(net, val_loader)

        _train_loss.append(loss_to_show / (step + 1))
        _test_loss.append(v_test_loss)
        _pixel_acc.append(v_acc)
        _mIoU.append(v_mean_iou)
        _sc_metrix.append([v_prec, v_recall, v_iou])
        _sscIoU.append(v_ssc_iou)

        print('Validate with TSDF:epoch {}, p {:.1f}, r {:.1f}, IoU {:.1f}'.format(epoch + 1, v_prec * 100.0,
                                                                                   v_recall * 100.0, v_iou * 100.0))
        if args.task == "ssc":
            print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(v_acc * 100.0, v_mean_iou * 100.0,
                                                                         v_ssc_iou * 100.0))
            is_best = v_mean_iou > best_miou
            best_miou = max(v_mean_iou, best_miou)
        else:
            is_best = v_iou > best_miou
            best_miou = max(v_iou, best_miou)
        scheduler.step(loss_to_show)
        # scheduler.step()
        # ---- Save Checkpoint

        state = {'state_dict': net.state_dict()}
        torch.save(state, cp_filename)
        if is_best:
            if args.task == "ssc":
                print('Yeah! Got better mIoU {}% in epoch {}. State saved'.format(100.0 * v_mean_iou, epoch + 1))
            else:
                print('Yeah! Got better IoU {}% in epoch {}. State saved'.format(100.0 * v_iou, epoch + 1))
            torch.save(state, cp_best_filename)  # Save Checkpoint


        epoch = epoch + 1
        if epoch % args.vis_every == 0 and best_miou > 0.3:
            try:
                vis_res(net, epoch+1)
            except:
                pass
        save_logs(_train_loss, _test_loss, _pixel_acc, np.array(_sc_metrix).T, _mIoU, np.array(_sscIoU).T)


def split_test_points(model, test_points, rgb, depth, p, batch_test=12960):
    num_points = test_points.shape[1]
    iters = int(num_points / batch_test)
    pred = []
    for i in range(iters):
        y_pred = model(x_depth=depth, x_rgb=rgb, p=p,
                               query_points=test_points[:,i*batch_test:(i+1)*batch_test,:])
        pred.append(y_pred)
    
    if iters * batch_test < num_points:
        y_pred = model(x_depth=depth, x_rgb=rgb, p=p,
                                query_points=test_points[:,iters*batch_test:,:])
    
        pred.append(y_pred)
    return torch.cat(pred,dim=1)

def validate_on_test_dataset(model, data_loader, save_ply=False):
    """
    Evaluate on validation set.
        model: network with parameters loaded
        date_loader: TEST mode
    """
    model.eval()  # switch to evaluate mode.
    val_acc, val_p, val_r, val_iou, test_losses = 0.0, 0.0, 0.0, 0.0, 0.0
    _C = 12 if args.task == "ssc" else 2
    val_cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    val_iou_ssc = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    count = 0
    loss_func = torch.nn.CrossEntropyLoss(weight=config.class_weights, ignore_index=255).cuda()
    with torch.no_grad():
        # ---- STSDF  depth, input, target, position, _
        for step, (rgb, depth, tsdf, test_points, y_true, target_lr, nonempty, position, filename) in tqdm(
                enumerate(data_loader), desc='Validating', unit='frame'):
            var_x_depth = depth.float().cuda()
            position = position.long().cuda()
            test_points = test_points.float().cuda()
            var_x_rgb = rgb.float().cuda()
            var_x_tsdf = tsdf.float().cuda()

            try:
                y_pred = model(x_depth=var_x_depth, x_rgb=var_x_rgb, p=position,
                            query_points=test_points)
            except:
                y_pred = split_test_points(model, test_points, var_x_rgb, var_x_depth, position)
            _y_true = y_true.long().contiguous()
            _y_true = _y_true.view(-1).cuda()
            _y_pred = y_pred.contiguous()
            _y_pred = _y_pred.view(-1, _C)

            if args.task == "ssc":
                loss_ssc = loss_func(_y_pred, _y_true)
                loss_geo = geometry_loss(_y_pred, _y_true)
                loss = loss_ssc + 2 * loss_geo
            elif args.task == "sc":
                loss_geo = geometry_loss(_y_pred, _y_true)
                loss = loss_geo
            test_loss = loss.item()
            test_losses += test_loss

            y_pred = y_pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy
            y_true = y_true.cpu().data.numpy()  # torch tensor to numpy
            b,c,_ = y_pred.shape
            #print(nonempty.shape)
            nonempty = nonempty.numpy().transpose((0,2,3,1)).reshape((b,-1)) # b, c, D, H, W -> b, c, H, W, D

            p, r, iou, acc, iou_sum, cnt_class = validate_on_batch(y_pred, y_true, nonempty)
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


if __name__ == '__main__':
    train()
