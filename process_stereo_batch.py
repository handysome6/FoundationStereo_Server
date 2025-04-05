##################################################
# Please make sure create your own custom_config.yaml
# Make sure the intrinsic_file and scale is provided
##################################################


from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from stereo_service import StereoService

# THIS IS THE NEW VERSION, MUCH FASTER
def process_stereo_batch(rect_img_dir, output_dir, config_path):
    """
    This is the new version, much faster, using the stereo_service.py
    Args:
        rect_img_dir: 包含已校正图像的文件夹路径
        output_dir: 输出文件夹路径
        config_path: 配置文件路径
    """
    stereo_service = StereoService(config_path=config_path)
    stereo_service.preheat()

    # PROCESS all images in a folder
    ROOT_DIR = Path(rect_img_dir)
    batch_save_dir = output_dir
    intrinsic_file = stereo_service.config['intrinsic_file']
    assert intrinsic_file is not None, "intrinsic_file is not provided"
    
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


if __name__ == "__main__":
    process_stereo_batch("/home/andy/DCIM/0327_1/rectified/", "./output/0327_1", "custom_config.yaml")




# THIS IS THE OLD VERSION, VERY SLOW
def process_stereo_batch_old(rect_img_dir, output_dir, args):
    """
    处理文件夹中的所有立体图像对
    
    Args:
        rect_img_dir: 包含已校正图像的文件夹路径
        output_dir: 输出文件夹路径
        args: 处理参数
    """
    from scripts.run_demo import process_stereo_images

    ROOT_DIR = Path(rect_img_dir)
    batch_save_dir = output_dir
    
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
        
        # 更新处理参数
        args.left_file = str(left_img_file)
        args.right_file = str(right_img_file)
        args.out_dir = f"{batch_save_dir}/{id}"
        
        # 处理当前图像对
        process_stereo_images(args)

# if __name__ == "__main__":
#     args = OmegaConf.load('custom_config.yaml')
#     process_stereo_batch_old("/home/andy/DCIM/0327_handheld_nuclear/good", "./output/0327_good", args)

