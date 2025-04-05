from PIL import Image
import numpy as np
from .y_diff import ImagePairAnalyzer
import shutil
from pathlib import Path
from loguru import logger


def adjust_image_coordinates(input_path, output_path, x_offset, y_offset):
    """
    调整图像像素坐标并保持原始尺寸，超出边界的像素将被删除（不会循环回来）
    
    参数:
        input_path (str): 输入图像的路径
        output_path (str): 输出图像的保存路径
        x_offset (int): x坐标的偏移量
        y_offset (int): y坐标的偏移量
    """
    # 打开原始图像
    original_img = Image.open(input_path)
    width, height = original_img.size
    
    # 创建一个新的空白图像，与原图尺寸相同
    adjusted_img = Image.new(original_img.mode, (width, height), (0, 0, 0, 0))
    
    # 将原图转换为numpy数组以便处理
    original_array = np.array(original_img)
    
    # 创建调整后图像的numpy数组
    adjusted_array = np.zeros_like(original_array)
    
    # 应用坐标偏移
    for y in range(height):
        for x in range(width):
            # 计算新坐标
            new_x = int(x + x_offset)
            new_y = int(y + y_offset)
            
            # 只复制在图像范围内的像素
            if 0 <= new_x < width and 0 <= new_y < height:
                adjusted_array[new_y, new_x] = original_array[y, x]
    
    # 将numpy数组转换回图像
    adjusted_img = Image.fromarray(adjusted_array)
    
    # 保存结果
    adjusted_img.save(output_path)
    
    print(f"调整后的图像已保存至 {output_path}")


def adjust_image_pairs(left_path, right_path, output_dir):
    """
    调整左右图像对的坐标
    
    参数:
        left_path (str): 左图像的路径
        right_path (str): 右图像的路径
        output_dir (str): 输出图像的保存路径
    """


def adjust_image_folder(input_dir, good_dir, bad_dir):
    """
    调整图像文件夹中所有图像的坐标
    
    参数:
        input_dir (str): 输入图像文件夹的路径
        output_dir (str): 输出图像文件夹的路径
    """
    # 遍历输入文件夹中的所有图像
    ROOT_DIR = Path(input_dir)
    GOOD_DIR = Path(good_dir)
    BAD_DIR = Path(bad_dir)
    # make sure the good and bad dirs exist
    GOOD_DIR.mkdir(parents=True, exist_ok=True)
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 初始化计数器
    good_pairs = 0
    bad_pairs = 0
    
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
        
        analyzer = ImagePairAnalyzer(left_img_file, right_img_file, detector_name="AKAZE", matcher="BF")
        median_diff = analyzer.get_filtered_median(c=0.10)
        print(f"median: {median_diff}")

        if abs(median_diff) < 5:
            # just copy the left and right to good
            shutil.copy(left_img_file, GOOD_DIR / left_img_file.name)
            shutil.copy(right_img_file, GOOD_DIR / right_img_file.name)
            good_pairs += 1
        else:
            # adjust the right image and copy to bad
            adjust_image_coordinates(right_img_file, BAD_DIR / right_img_file.name, 0, -median_diff)
            # copy the left image to the output directory
            shutil.copy(left_img_file, BAD_DIR / left_img_file.name)
            bad_pairs += 1
    
    # 输出统计信息
    logger.info(f"处理完成！统计信息：")
    logger.info(f"好的图像对数量: {good_pairs}")
    logger.info(f"需要调整的图像对数量: {bad_pairs}")
    logger.info(f"总图像对数量: {good_pairs + bad_pairs}")



# 测试
if __name__ == "__main__":
#     # single image
#     adjust_image_coordinates(
#         input_path="/home/andy/DCIM/0327_1/rectified/D_87522380771.jpg",
#         output_path="/home/andy/DCIM/0327_1/rectified/D_87522380771_adjusted.jpg",
#         x_offset=0,
#         y_offset=9 // 0.35)
    
    # image pairs
    adjust_image_folder(
        input_dir="/home/andy/DCIM/0327_handheld_nuclear/rectified",
        good_dir="/home/andy/DCIM/0327_handheld_nuclear/good",
        bad_dir="/home/andy/DCIM/0327_handheld_nuclear/bad")