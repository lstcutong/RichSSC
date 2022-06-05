import open3d as o3d
import numpy as np
import sys
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import cv2
import os
import time
from scipy.spatial.transform import Rotation as R
import mcubes
import trimesh
from skimage.measure import marching_cubes_lewiner
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

def convert_voxel_into_open3dmesh(voxels, size=0.2):
    '''

    :param voxels: # [W, H, D, 3], [0, 0, 0] represents empty space
    :param size: [0,1] reprensents each voxel's size
    :return:
    '''
    assert size >= 0 and size <= 1, "size should be in [0, 1]"
    W, H, D = voxels.shape[0:3]
    X, Y, Z = np.meshgrid(np.arange(0, W), np.arange(0, H), np.arange(0, D), indexing="ij")
    _l = np.max(np.array([W, H, D]))
    true_size = 2 / _l

    grid_center = np.array([X + 0.5 , Y + 0.5, Z + 0.5]).transpose()
    grid_center = grid_center.reshape((-1, 3))
    # to -1, 1
    grid_center = grid_center / np.max(grid_center) * 2 - 1
    voxels = voxels.reshape((-1, 3))

    non_empty_index = np.where(voxels.sum(1) != 0)
    grid_center = grid_center[non_empty_index]
    voxels = voxels[non_empty_index]

    all_vertex, all_triangle, all_vertex_color, all_normal = [], [], [], []
    for i in range(len(grid_center)):
        box = o3d.geometry.TriangleMesh().create_box(width=true_size*size, height=true_size*size, depth=true_size*size)
        box.translate(grid_center[i])
        box.compute_vertex_normals()
        vertex = np.asarray(box.vertices)
        triangle = np.asarray(box.triangles) + i * 8
        vertex_colors = np.repeat(voxels[i][None, :], 8, axis=0) / 255
        normals = np.asarray(box.vertex_normals)

        all_vertex.append(vertex)
        all_triangle.append(triangle)
        all_vertex_color.append(vertex_colors)
        all_normal.append(normals)

    all_vertex = np.row_stack(all_vertex)
    # move scene center to (0,0,0)
    center_of_scene = np.mean(all_vertex, axis=0)
    all_vertex = all_vertex - center_of_scene

    all_triangle = np.row_stack(all_triangle)
    all_vertex_color = np.row_stack(all_vertex_color)
    all_normal = np.row_stack(all_normal)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(all_vertex)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(all_triangle)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_color)
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(all_normal)
    return o3d_mesh


def convert_voxel_into_iossurface(voxel, smooth=False, thresh=1):
    if smooth:
        voxel = mcubes.smooth(voxel, method="constrained")
    #vt, fa = mcubes.marching_cubes(voxel, thresh)
    vt, fa, _, _, = marching_cubes_lewiner(voxel, thresh)

    mesh = trimesh.Trimesh(vertices=vt, faces=fa)
    return mesh


def custom_draw_geometry_with_rotation_wo_img(geo):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(30.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([geo], rotate_view)


def custom_draw_geometry_with_rotation_wo_blocking(geo, save_folder, height=480, width=640):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, height=height, width=width, top=50, left=50)
    vis.add_geometry(geo)
    #vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(1))
    opt = vis.get_render_option()
    opt.point_size = 0.1
    ctr = vis.get_view_control()
    ctr.scale(10)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    rot44 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    for i in range(0, 360, 10):
        rad = np.radians(10)
        rot33 = R.from_rotvec(np.array([0, rad, 0])).as_matrix()
        rot44[0:3,0:3] = rot33
        geo.transform(rot44)
        vis.update_geometry(geo)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(save_folder, "{:03d}.png".format(i)))
    vis.destroy_window()


if __name__ == '__main__':
    pass
