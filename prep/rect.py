import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .model import CameraModel


class StereoRectify():
    def __init__(self, camera, operation_folder = None) -> None:
        """Construct rectifier.
        camera: calibrated CameraModel object
        operation_folder: Path object, folder containing 'scenes/'
        """
        self.camera = camera
        self.width = self.camera.image_size[0]
        self.height = self.camera.image_size[1]

        # init Q and maps
        self.Q = None
        self.leftMapX = self.leftMapY = None
        self.rightMapX = self.rightMapY = None

        # folder Path object
        if operation_folder is not None:
            self.operation_folder = operation_folder
            self.scenes_folder = self.operation_folder / 'scenes'
            self.rectify_folder = self.operation_folder / 'rectified'
            self.data_folder = self.operation_folder / 'calibration_data'
            self.camera_folder = self.operation_folder / 'camera_model'


    def rectify_camera(self, roi_ratio = 0, new_image_ratio = 1):
        """
        Switch to call diff rectify method 
        roi_ratio: Determine how much black edge is preserved
                    roi_ratio = 0: None black area is preserved
                    roi_ratio = 1: all black area is preserved
        new_image_ratio: Determine the new imagesize 
        """
        # rectify parameters
        roi_ratio = roi_ratio
        newImageSize = np.array(self.camera.image_size) * new_image_ratio
        newImageSize = np.array(newImageSize, dtype=np.int32)

        if not self.camera.is_calibrated():
            print("No calib_data found. \nPlease calib camera before rectify")
            exit()
        if self.camera.is_fisheye:
            self._stereo_rectify_fisheye(roi_ratio, newImageSize)
        else:
            self._stereo_rectify_vanilla(roi_ratio, newImageSize)

    def is_rectified(self):
        """Check if this rectifier is rectified"""
        if  self.Q is None or\
            self.leftMapX  is None or self.leftMapY  is None or \
            self.rightMapX is None or self.rightMapY is None:
            return False
        else:
            return True


    def rectify_image(self, img_left=None, img_right=None, sbs_img=None):
        """ 
        Rectify single sbs image using maps
        img_left: left img
        img_right: right img
        sbs_img: single sbs image
        """
        # ensure rectify parameters exist
        if not self.is_rectified():
            print("Rectifier not rectified, rectifying first...")
            self.rectify_camera()
        
        if img_left is not None and img_right is not None:
            img_left = img_left
            img_right = img_right
        elif sbs_img is not None:
            # split
            img_left = sbs_img [:,          0:   self.width]
            img_right = sbs_img [:, self.width: 2*self.width]
        else:
            raise Exception("At least one pair of img should be provided. "
                "Either sbs_img or img_left/right.")
        
        # rectify the given image
        left_rect = cv2.remap(
            img_left, self.leftMapX, self.leftMapY, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        right_rect = cv2.remap(
            img_right, self.rightMapX, self.rightMapY, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return left_rect, right_rect

    def rectify_samples(self):
        """
        Rectify sample sbs images from "operation_folder / scenes"
        Save to "operation_folder / rectified"
        """
        # mkdir if folder not exist
        self.rectify_folder.mkdir(parents=False, exist_ok=True)

        # iterate all sbs images
        scenes_imgs = list(self.scenes_folder.iterdir())
        i = 0
        for img_path in tqdm(scenes_imgs, desc="Rectifying".ljust(10)):
            i += 1
            sbs_img = cv2.imread(str(img_path))
            imgL, imgR = self.rectify_image(sbs_img = sbs_img)
            left_name  = f"rectify_{str(i).zfill(2)}_left.jpg"
            right_name = f"rectify_{str(i).zfill(2)}_right.jpg"
            cv2.imwrite(str(self.rectify_folder / left_name), imgL)
            cv2.imwrite(str(self.rectify_folder / right_name), imgR)
        print("Rectify images done.")


    def _stereo_rectify_vanilla(self, alpha, newImageSize):  
        """
        Compute rectify map in Vanilla approach
        """
        print("Vanilla rectifying...")
        # calculate rectify matrices using calibration param
        R1, R2, P1, P2, Q, ROI1, ROI2 = \
            cv2.stereoRectify(
                self.camera.cm1, self.camera.cd1, 
                self.camera.cm2, self.camera.cd2, 
                self.camera.image_size, 
                self.camera.R, self.camera.T,
                alpha=alpha,
                newImageSize=newImageSize,
            )
        
        self.Q = Q
        # create map for rectification
        self.leftMapX, self.leftMapY  = cv2.initUndistortRectifyMap(
            self.camera.cm1, self.camera.cd1, R1, P1, newImageSize, cv2.CV_16SC2
        )
        self.rightMapX, self.rightMapY= cv2.initUndistortRectifyMap(
            self.camera.cm2, self.camera.cd2, R2, P2, newImageSize, cv2.CV_16SC2
        )
        print("Calculate map done.")
        print(self.rightMapX.shape)
        print()
        print()
        print()
        print()

    def _stereo_rectify_fisheye(self, alpha, newImageSize):
        """
        Compute rectify map in Fisheye approach - TODO
        """
        pass


def rectify_stereo_images(camera_model_path, raw_img_dir, output_dir):
    """
    处理双目相机图像的矫正
    
    Args:
        camera_model_path (str or Path): 相机模型文件的路径
        raw_img_dir (str or Path): 原始图像所在的目录
        output_dir (str or Path): 输出目录路径
    """
    # 确保输入路径是Path对象
    camera_model_path = Path(camera_model_path)
    raw_img_dir = Path(raw_img_dir)
    
    # 加载相机模型并创建矫正器
    camera = CameraModel.load_model(camera_model_path)
    rectifier = StereoRectify(camera)
    rectifier.rectify_camera()

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 获取所有左侧图像路径
    left_images = list(raw_img_dir.glob("A_*.jpg"))
    
    # 处理所有图像对，添加进度条
    for left_path in tqdm(left_images, desc="正在矫正图像", ncols=80):
        right_path = left_path.with_name(left_path.name.replace("A_", "D_"))
        
        # 读取图像
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        # 验证图像
        assert left_img is not None, f"无法读取左图像: {left_path}"
        assert right_img is not None, f"无法读取右图像: {right_path}"
        assert left_img.shape == right_img.shape, f"左右图像尺寸不一致: {left_path} {left_img.shape} != {right_path} {right_img.shape}"

        # 矫正图像
        left_rect, right_rect = rectifier.rectify_image(left_img, right_img)

        # 保存矫正后的图像
        left_save_path = output_dir / left_path.name
        right_save_path = output_dir / right_path.name
        cv2.imwrite(str(left_save_path), left_rect)
        cv2.imwrite(str(right_save_path), right_rect)

if __name__ == "__main__":
    from icecream import ic
    model_path = r"/home/andy/DCIM/0327_handheld_nuclear/camera_model.json"
    raw_dir = r"/home/andy/DCIM/0327_handheld_nuclear/"
    output_dir = r"/home/andy/DCIM/0327_handheld_nuclear/rectified"
    rectify_stereo_images(model_path, raw_dir, output_dir)
