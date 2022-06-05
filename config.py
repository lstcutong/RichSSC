import numpy as np
import torch

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'nyu':
            # folder that contains dataset/.
            return {'train': '/home/magic/Datasets/nyu-tsdf/NYUtrain_npz',
                    'val': '/home/magic/Datasets/nyu-tsdf/NYUtest_npz',}

        elif dataset == 'nyucad':
            return {'train': '/home/magic/Datasets/nyu-tsdf/NYUCADtrain_npz',
                    'val': '/home/magic/Datasets/nyu-tsdf/NYUCADtest_npz'}

        # debug
        elif dataset == 'debug':
            return {'train': '/home/magic/Datasets/nyu-tsdf/NYUCADDebug_npz',
                    'val': '/home/magic/Datasets/nyu-tsdf/NYUCADDebug_npz',}
        elif dataset == "suncg":
            return {'train': '/home/magic/Datasets/suncg/suncg_npz',
                    'val': '/home/magic/Datasets/suncg/suncg_test_npz',}
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


cls_name = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furn", "objects"]
# ssc: color map
colorMap = np.array([[0, 0, 0],    # 0 empty, free space
                     [214,  38, 40],    # 1 ceiling
                     [43, 160, 4],      # 2 floor
                     [158, 216, 229],   # 3 wall
                     [114, 158, 206],   # 4 window
                     [204, 204, 91],    # 5 chair  new: 180, 220, 90
                     [255, 186, 119],   # 6 bed
                     [147, 102, 188],   # 7 sofa
                     [30, 119, 181],    # 8 table
                     [188, 188, 33],    # 9 tvs
                     [255, 127, 12],    # 10 furn
                     [196, 175, 214],   # 11 objects
                     [153, 153, 153],     # 12 Accessible area, or label==255, ignore
                     ]).astype(np.int32)

# ###########################################################################################

class_weights = torch.FloatTensor([0.8, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1])
class_weights_geo = torch.FloatTensor([0.8, 1])

