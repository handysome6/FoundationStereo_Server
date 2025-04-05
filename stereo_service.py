import os
import yaml
import torch
import numpy as np
import cv2
import imageio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Union, Optional
from omegaconf import OmegaConf
import open3d as o3d
from loguru import logger

from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo
from Utils import depth2xyzmap, toOpen3dCloud, set_logging_format, set_seed, timeit

class StereoService:
    """
    Providing class based interface for stereo service, 
    including preheat, process_stereo_pair, _generate_point_cloud
    """
    def __init__(self, config_path: str):
        """
        初始化立体视觉服务
        Args:
            config_path: YAML配置文件的路径
        """
        set_logging_format()
        set_seed(0)
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._initialize_model()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        return OmegaConf.load(config_path)
    
    def _initialize_model(self):
        """初始化模型并加载检查点"""
        torch.autograd.set_grad_enabled(False)
        self.model = FoundationStereo(self.config)
        ckpt_path = self.config.get('ckpt_dir')
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, weights_only=False)
            logging.info(f"ckpt global_step:{checkpoint['global_step']}, epoch:{checkpoint['epoch']}")
            self.model.load_state_dict(checkpoint['model'])
        self.model.cuda()
        self.model.eval()
    
    def preheat(self, image_size=(1280, 720)):
        """
        使用随机生成的图像预热模型
        Args:
            image_size: 生成图像的大小，格式为(高度, 宽度)
        """
        logging.info("开始使用随机图像预热模型...")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory(prefix='stereo_preheat_') as temp_dir:
            logging.info(f"创建临时目录: {temp_dir}")
            
            # 生成5对随机图像进行预热
            for i in range(5):
                try:
                    # 生成随机图像对
                    # 左图像：随机生成的RGB图像
                    left_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                    # 右图像：对左图像进行轻微变换以模拟视差
                    right_img = np.roll(left_img, shift=np.random.randint(-20, -5), axis=1)
                    right_img = cv2.GaussianBlur(right_img, (3, 3), 1.0)
                    
                    # 创建临时文件
                    with tempfile.NamedTemporaryFile(suffix='.png', prefix=f'left_{i}_', dir=temp_dir) as left_temp, \
                         tempfile.NamedTemporaryFile(suffix='.png', prefix=f'right_{i}_', dir=temp_dir) as right_temp:
                        
                        # 保存图像到临时文件
                        cv2.imwrite(left_temp.name, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(right_temp.name, cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
                        
                        # 处理图像对
                        custom_cfg = OmegaConf.create({
                            'out_dir': temp_dir,
                            'scale': 1,
                            'get_pc': 0
                        })
                        self.process_stereo_pair(
                            left_temp.name, 
                            right_temp.name,
                            custom_cfg
                        )
                        
                    logging.info(f"完成第 {i+1}/10 次预热")
                    
                except Exception as e:
                    logging.error(f"预热过程中出错: {str(e)}")
                    continue
            
            logging.info("模型预热完成")
    
    @timeit
    def process_stereo_pair(self, 
                          left_file: str,
                          right_file: str,
                          custom_args: Optional[Union[Dict, OmegaConf]] = None) -> str:
        """处理立体图像对"""
        logger.info(f"开始处理立体图像对: {left_file} 和 {right_file}")
        
        # 使用OmegaConf合并配置
        current_config = OmegaConf.merge(self.config, custom_args if custom_args else {})
        logger.debug(f"当前配置: {current_config}")
            
        # 创建输出目录
        output_dir = os.path.join(
            current_config.get('out_dir', './outputs'),
            Path(left_file).stem
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")
        
        # 保存当前配置到输出目录
        with open(f'{output_dir}/args.yaml', 'w') as f:
            OmegaConf.save(current_config, f)
        logger.debug("配置已保存到输出目录")
        
        # 读取和处理图像
        logger.info("开始读取图像...")
        img0 = imageio.imread(left_file)
        img1 = imageio.imread(right_file)
        logger.info(f"图像读取完成，左图像尺寸: {img0.shape}，右图像尺寸: {img1.shape}")
        
        scale = current_config.get('scale', 1.0)
        assert scale <= 1, "scale must be <=1"
        
        if scale != 1:
            logger.info(f"应用缩放因子: {scale}")
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
            logger.info(f"缩放后图像尺寸: {img0.shape}")
        
        # 保存处理后的输入图像
        logger.info("保存处理后的输入图像...")
        imageio.imwrite(f'{output_dir}/img0.png', img0)
        imageio.imwrite(f'{output_dir}/img1.png', img1)
        
        H, W = img0.shape[:2]
        img0_ori = img0.copy()
        
        # 准备模型输入
        logger.info("准备模型输入...")
        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        
        # 模型推理
        logger.info("开始模型推理...")
        with torch.amp.autocast('cuda'):
            if not current_config.get('hiera', 0):
                logger.debug("使用标准推理模式")
                disp = self.model.forward(img0, img1, 
                                        iters=current_config.get('valid_iters', 32), 
                                        test_mode=True)
            else:
                logger.debug("使用分层推理模式")
                disp = self.model.run_hierachical(img0, img1, 
                                                iters=current_config.get('valid_iters', 32), 
                                                test_mode=True, 
                                                small_ratio=0.5)
                
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        logger.info("模型推理完成")
        
        # 保存视差图
        logger.info("保存视差图...")
        np.save(f'{output_dir}/disp.npy', disp)
        
        # 检查是否需要生成点云
        if current_config.get('get_pc', 0):
            logger.info("开始生成点云...")
            # 检查是否有内参文件provided and exists
            if 'intrinsic_file' in current_config and Path(current_config.get('intrinsic_file')).exists():
                self._generate_point_cloud(disp, img0_ori, current_config, output_dir)
                logger.info("点云生成完成")
            else:
                logger.error("需要生成点云但未提供内参文件")
                return None
        else:
            logger.info("跳过点云生成")
            
        logger.info(f"处理完成，输出目录: {output_dir}")
        return output_dir
    
    def _generate_point_cloud(self, disp, img0_ori, config, output_dir):
        """生成并保存点云"""
        # 读取相机内参
        if not os.path.exists(config.get('intrinsic_file', '')):
            logging.error(f"内参文件不存在: {config.get('intrinsic_file')}")
            return
            
        with open(config.get('intrinsic_file'), 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
            
        scale = config.get('scale', 1.0)
        K[:2] *= scale
        
        if config.get('remove_invisible', 1):
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf
            
        depth = K[0,0] * baseline / disp
        np.save(f'{output_dir}/depth_meter.npy', depth)
        
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
        
        # 创建掩码来标识不合格的点
        invalid_mask = ~((np.asarray(pcd.points)[:,2]>0) & 
                        (np.asarray(pcd.points)[:,2]<=config.get('z_far', 10)))
        
        # 将不合格点的坐标设置为(0,0,0)
        points = np.asarray(pcd.points)
        points[invalid_mask] = np.array([0,0,0])
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 保存原始点云
        o3d.io.write_point_cloud(f'{output_dir}/cloud.ply', pcd)
        
        # 如果需要去噪
        if config.get('denoise_cloud', 0):
            logging.info("denoise point cloud...")
            cl, ind = pcd.remove_radius_outlier(
                nb_points=config.get('denoise_nb_points', 30),
                radius=config.get('denoise_radius', 0.03)
            )
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(f'{output_dir}/cloud_denoise.ply', inlier_cloud)
    
    def __call__(self, 
                left_file: str,
                right_file: str,
                custom_args: Optional[Union[Dict, OmegaConf]] = None) -> str:
        """方便的调用接口，直接处理图像对"""
        return self.process_stereo_pair(left_file, right_file, custom_args)
    

if __name__ == "__main__":
    stereo_service = StereoService(config_path='server_config.yaml')
    stereo_service.preheat()

    # PROCESS all images in a folder
    ROOT_DIR = Path("/home/andy/DCIM/0327_1/rectified/")
    batch_save_dir = "output/0327_1"
    intrinsic_file = "/home/andy/workspace/FoundationStereo/assets/K_477module.txt"
    
    # 获取图像后缀名（从第一个A_开头的文件）
    img_suffix = list(ROOT_DIR.glob("A_*"))[0].suffix
    
    # 遍历所有左图像
    for left_img_file in ROOT_DIR.glob(f"A_*{img_suffix}"):
        # 根据左图像文件名构造右图像文件名
        id = left_img_file.stem[2:]  # 去掉"A_"前缀
        right_img_file = ROOT_DIR / f"D_{id}{img_suffix}"
        
        if not right_img_file.exists():
            logger.warning(f"找不到对应的右图像: {right_img_file}")
            continue
            
        logger.info(f"处理图像对: {left_img_file.name} - {right_img_file.name}")
        
        # 处理图像对
        custom_cfg = OmegaConf.create({
            'out_dir': batch_save_dir,
            'scale': 0.35,
            'get_pc': 1,
            'intrinsic_file': intrinsic_file
        })
        stereo_service.process_stereo_pair(
            left_img_file, 
            right_img_file,
            custom_cfg
        )