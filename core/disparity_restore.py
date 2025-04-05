import numpy as np
import cv2
from icecream import ic


class DisparityRestore:
    """用于恢复真实视差值的类
    
    该类可以根据原始校正后的左右图像和缩放后的视差图(npy)来恢复真实视差值。
    缩放参数会根据原始图像和视差图的尺寸自动计算。
    
    Attributes:
        left_img (np.ndarray): 原始校正后的左图像
        right_img (np.ndarray): 原始校正后的右图像
        scaled_disp (np.ndarray): 缩放后的视差图
        scale (float): 自动计算的缩放参数
    """
    
    def __init__(self, left_img, right_img, scaled_disp):
        """初始化DisparityRestore类
        
        Args:
            left_img (np.ndarray): 原始校正后的左图像，shape为(H,W,3)
            right_img (np.ndarray): 原始校正后的右图像，shape为(H,W,3)
            scaled_disp (np.ndarray): 缩放后的视差图，shape为(H',W')
        """
        self.left_img = left_img
        self.right_img = right_img
        self.scaled_disp = scaled_disp
        
        # 获取原始图像尺寸
        self.orig_h, self.orig_w = left_img.shape[:2]
        
        # 获取缩放后的视差图尺寸
        self.scaled_h, self.scaled_w = scaled_disp.shape
        
        # 计算缩放参数
        self.scale = min(self.scaled_h / self.orig_h, self.scaled_w / self.orig_w)
        assert self.scale <= 1, "视差图尺寸不应大于原始图像尺寸"
        ic(self.scale)
        
    def restore_disparity(self, x, y):
        """恢复指定像素位置的真实视差值
        
        Args:
            x (int): 像素x坐标
            y (int): 像素y坐标
            
        Returns:
            float: 恢复后的真实视差值
        """
        # 将原始坐标映射到缩放后的坐标
        scaled_x = int(x * self.scale)
        scaled_y = int(y * self.scale)
        
        # 检查坐标是否在有效范围内
        if scaled_x < 0 or scaled_x >= self.scaled_w or scaled_y < 0 or scaled_y >= self.scaled_h:
            return np.inf
            
        # 获取缩放后的视差值
        scaled_disp_value = self.scaled_disp[scaled_y, scaled_x]
        
        # 如果视差值为无效值，返回无穷大
        if scaled_disp_value == np.inf or np.isnan(scaled_disp_value):
            return np.inf
            
        # 恢复真实视差值
        real_disp = scaled_disp_value / self.scale
        
        return real_disp
        
    def restore_full_disparity(self):
        """恢复整个视差图的真实视差值
        
        Returns:
            np.ndarray: 恢复后的完整视差图，shape为(H,W)
        """
        # 创建与原始图像相同大小的视差图
        real_disp = np.zeros((self.orig_h, self.orig_w), dtype=np.float32)
        
        # 对每个像素进行视差恢复
        for y in range(self.orig_h):
            for x in range(self.orig_w):
                real_disp[y, x] = self.restore_disparity(x, y)
                
        return real_disp 
    

if __name__ == "__main__":
    from icecream import ic
    from pathlib import Path
    import cv2

    left_img = cv2.imread(str(Path("test_img") / "A_10211647201283.jpg"))
    right_img = cv2.imread(str(Path("test_img") / "D_10211647201283.jpg"))
    scaled_disp = np.load(str(Path("test_outputs") / "disp.npy"))
    ic(left_img.shape, right_img.shape, scaled_disp.shape)

    restorer = DisparityRestore(left_img, right_img, scaled_disp)
    real_disp = restorer.restore_full_disparity()
    ic(real_disp.shape)

    cv2.imwrite(str(Path("test_outputs") / "real_disp.jpg"), real_disp)