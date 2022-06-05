import numpy as np
import torch
import torch.nn.functional as F

def downsample_vox_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (voxel_size[0] // ds, voxel_size[1] // ds, voxel_size[2] // ds)  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0]*small_size[1]*small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])
        # z, y, x = np.unravel_index(i, small_size)  # 速度更慢了
        # print(x, y, z)

        label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
        label_bin = label_i.flatten()  # faltten 返回的是真实的数组，需要分配新的内存空间
        # label_bin = label_i.ravel()  # 将多维数组变成 1维数组，而ravel 返回的是数组的视图

        # zero_count_0 = np.sum(label_bin == 0)
        # zero_count_255 = np.sum(label_bin == 255)
        zero_count_0 = np.array(np.where(label_bin == 0)).size  # 要比sum更快
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
            label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale

def downsample_vox_labelv2(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = int(voxel_size[0] // ds), int(voxel_size[1] // ds), int(voxel_size[2] // ds)  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
      # threshold
    s01 = small_size[0] * small_size[1]
    #label_i = np.zeros((int(ds), int(ds), int(ds)), dtype=np.int32)
    #print(small_size)
    last_volume_x, last_volume_y, last_volume_z = 0, 0, 0
    for i in range(small_size[0]*small_size[1]*small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])
        # z, y, x = np.unravel_index(i, small_size)  # 速度更慢了
        # print(x, y, z)
        #print(x,y,z)

        label_i = label[int(x * ds):int((x + 1) * ds), int(y * ds):int((y + 1) * ds), int(z * ds):int((z + 1) * ds)]

        sx, sy, sz =label_i.shape
        empty_t = 0.95 * sx * sy * sz
        #print(label_i)
        label_bin = label_i.flatten()  # faltten 返回的是真实的数组，需要分配新的内存空间
        # label_bin = label_i.ravel()  # 将多维数组变成 1维数组，而ravel 返回的是数组的视图

        # zero_count_0 = np.sum(label_bin == 0)
        # zero_count_255 = np.sum(label_bin == 255)
        zero_count_0 = np.array(np.where(label_bin == 0)).size  # 要比sum更快
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
            #print(label_bin)
            label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
            #print(label_i_s)
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    #print(x,y,z)
    return label_downscale

def upsample_vox_label(label, upscale=4):
    n_vox = F.interpolate(torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float(), scale_factor=upscale,mode="nearest")
    return n_vox[0,0].int().numpy()


def rescale_vox_score(score, scale=1, mode="trilinear"):
    '''

    Args:
        score: torch.Tensor [batch, C, _, _, _]
        scale:

    Returns:

    '''
    if scale == 1:
        return score

    score = F.interpolate(score.float(), scale_factor=scale,mode=mode)
    return score



def downsample_tsdf(tsdf, downscale):
    if downscale == 1:
            return tsdf
        # TSDF_EMPTY = np.float32(0.001)
        # TSDF_SURFACE: 1, sign >= 0
        # TSDF_OCCLUD: sign < 0  np.float32(-0.001)
    ds = downscale
    small_size = (int(tsdf.shape[0] / ds), int(tsdf.shape[1] / ds), int(tsdf.shape[2] / ds))
    tsdf_downscale = np.ones(small_size, dtype=np.float32) * np.float32(0.001)  # init 0.001 for empty
    s01 = small_size[0] * small_size[1]
    tsdf_sr = np.ones((ds, ds, ds), dtype=np.float32)  # search region
    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])
        tsdf_sr[:, :, :] = tsdf[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
        tsdf_bin = tsdf_sr.flatten()
        # none_empty_count = np.array(np.where(tsdf_bin != TSDF_EMPTY)).size
        none_empty_count = np.array(np.where(np.logical_or(tsdf_bin <= 0, tsdf_bin == 1))).size
        if none_empty_count > 0:
            # surface_count  = np.array(np.where(stsdf_bin == 1)).size
            # occluded_count = np.array(np.where(stsdf_bin == -2)).size
            # surface_count = np.array(np.where(tsdf_bin > 0)).size  # 这个存在问题
            surface_count  = np.array(np.where(tsdf_bin == 1)).size
            # occluded_count = np.array(np.where(tsdf_bin < 0)).size
            # tsdf_downscale[x, y, z] = 0 if surface_count > occluded_count else np.float32(-0.001)
            tsdf_downscale[x, y, z] = 1 if surface_count > 2 else np.float32(-0.001)  # 1 or 0 ?
        # else:
        #     tsdf_downscale[x, y, z] = empty  # TODO 不应该将所有值均设为0.001
    return tsdf_downscale

def downsample_tsdfv2(tsdf, downscale):
    if downscale == 1:
        return tsdf
    # TSDF_EMPTY = np.float32(0.001)
    # TSDF_SURFACE: 1, sign >= 0
    # TSDF_OCCLUD: sign < 0  np.float32(-0.001)
    ds = downscale
    small_size = (int(tsdf.shape[0] / ds), int(tsdf.shape[1] / ds), int(tsdf.shape[2] / ds))
    tsdf_downscale = np.ones(small_size, dtype=np.float32) * np.float32(0.001)  # init 0.001 for empty
    s01 = small_size[0] * small_size[1]
    #tsdf_sr = np.ones((int(ds), int(ds), int(ds)), dtype=np.float32)  # search region
    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])
        tsdf_sr = tsdf[int(x * ds):int((x + 1) * ds), int(y * ds):int((y + 1) * ds), int(z * ds):int((z + 1) * ds)]
        tsdf_bin = tsdf_sr.flatten()
        # none_empty_count = np.array(np.where(tsdf_bin != TSDF_EMPTY)).size
        none_empty_count = np.array(np.where(np.logical_or(tsdf_bin <= 0, tsdf_bin == 1))).size
        if none_empty_count > 0:
            # surface_count  = np.array(np.where(stsdf_bin == 1)).size
            # occluded_count = np.array(np.where(stsdf_bin == -2)).size
            # surface_count = np.array(np.where(tsdf_bin > 0)).size  # 这个存在问题
            surface_count  = np.array(np.where(tsdf_bin == 1)).size
            # occluded_count = np.array(np.where(tsdf_bin < 0)).size
            # tsdf_downscale[x, y, z] = 0 if surface_count > occluded_count else np.float32(-0.001)
            tsdf_downscale[x, y, z] = 1 if surface_count > 2 else np.float32(-0.001)  # 1 or 0 ?
        # else:
        #     tsdf_downscale[x, y, z] = empty  # TODO 不应该将所有值均设为0.001
    return tsdf_downscale

def upsample_tsdf(tsdf, upscale):
    n_vox = F.interpolate(torch.from_numpy(tsdf).unsqueeze(0).unsqueeze(0).float(), scale_factor=upscale,mode="trilinear")
    return n_vox[0,0].int().numpy()

