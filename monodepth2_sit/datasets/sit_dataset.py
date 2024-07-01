import os
import numpy as np
import skimage.transform
import PIL.Image as pil
from sit_utils import generate_depth_map
from .mono_dataset import MonoDataset

def extract_scene_path(velo_filename):
    parts = velo_filename.split(os.sep)
    scene_path = os.path.join(parts[-5], parts[-4])
    return scene_path

class SITDataset(MonoDataset):
    """Superclass for different types of SIT dataset loaders"""
    def __init__(self, *args, **kwargs):
        super(SITDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1920, 1200)  # 모델 입력 해상도로 변경
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        if frame_index < 0:
            print(f"Invalid frame_index: {frame_index}, resetting to 0")
            frame_index = 0

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        image_path = self.get_image_path(folder, frame_index, side)
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
        try:
            color = self.loader(image_path)
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            raise e
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

class SITRAWDataset(SITDataset):
    """SIT dataset which loads the original velodyne depth maps for ground truth"""
    def __init__(self, *args, **kwargs):
        super(SITRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[str(side)]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        if frame_index < 0:
            print(f"Invalid frame_index: {frame_index}, resetting to 0")
            frame_index = 0

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        # Extract scene path
        scene_path = extract_scene_path(velo_filename)
        calib_path = os.path.join(self.data_path, scene_path, 'calib', '0.txt')

        depth_gt = generate_depth_map(self.data_path, velo_filename, self.side_map[str(side)])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
