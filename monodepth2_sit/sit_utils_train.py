from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format."""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def read_calib_file(path):
    """Read custom calibration file."""
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass
    return data

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices."""
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

# def generate_depth_map(base_dir, velo_filename, vel_depth=False):
def generate_depth_map(base_dir, velo_filename, cam=2, vel_depth=False):

    """Generate a depth map from velodyne data."""
    # Extract the scene name from the velo_filename path
    scene_name_parts = velo_filename.split(os.sep)
    scene_name = os.path.join(scene_name_parts[-5], scene_name_parts[-4])
    # scene_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(velo_filename))))

    # Construct the path to the calibration file
    calib_path = os.path.join(base_dir, scene_name, 'calib', '0.txt')

    # print("base_dir", base_dir)
    # print("velo_filename", velo_filename)
    # print("scene_name", scene_name)
    # print("calib_path", calib_path)

    # Load calibration data
    calib_data = read_calib_file(calib_path)

    # Print the keys to check if the required keys exist
    # print("calib_data keys:", calib_data.keys())

    # Extract intrinsic and extrinsic parameters for P3
    try:
        P_intrinsic = calib_data['P3_intrinsic'].reshape(3, 3)
        P_extrinsic = calib_data['P3_extrinsic'].reshape(3, 4)
    except KeyError as e:
        print(f"KeyError: {e} not found in calibration data")
        return None

    # Form the full 3x4 projection matrix
    P_velo2cam = np.vstack((P_extrinsic, np.array([0, 0, 0, 1.0])))

    # Load velodyne points and remove all points behind the image plane
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # Project the points to the camera
    velo_pts_im = np.dot(P_intrinsic, np.dot(P_velo2cam, velo.T)[:3, :]).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # Get image shape from intrinsic matrix
    im_shape = (int(P_intrinsic[1, 2] * 2), int(P_intrinsic[0, 2] * 2))

    # Check if in bounds
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # Project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # Find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


