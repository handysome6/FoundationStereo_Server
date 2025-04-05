#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from loguru import logger
import matplotlib
import platform

# 配置matplotlib支持中文显示
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    
    # 避免使用可能不存在的字体，使用更通用的设置
    if system == 'Darwin':  # macOS
        # 尝试使用macOS上常见的中文字体
        try:
            import matplotlib.font_manager as fm
            # 查找系统中存在的中文字体
            fonts = [f.name for f in fm.fontManager.ttflist 
                    if any(word in f.name.lower() for word in 
                           ['heiti', 'songti', 'pingfang', 'hiragino', 'arial unicode'])]
            
            if fonts:
                # 使用找到的第一个中文字体
                logger.info(f"使用字体: {fonts[0]}")
                matplotlib.rcParams['font.family'] = fonts[0]
            else:
                # 如果找不到特定中文字体，使用sans-serif族
                logger.info("未找到中文字体，使用sans-serif")
                matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
        except:
            # 出错时使用默认设置
            logger.warning("字体设置出错，使用默认设置")
            matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    
    elif system == 'Windows':
        # Windows系统
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
    else:
        # Linux系统
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif']
    
    # 正确显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False

# 在导入和配置完成后立即设置字体
setup_chinese_font()

MAX_FEATURES = 10000
MAX_MATCHES = 5000

class ImagePairAnalyzer:
    """图像对特征检测、匹配和错位统计分析"""
    
    def __init__(self, img1_path, img2_path, detector_name="SIFT", matcher="BF"):
        """
        初始化图像对分析器
        
        参数:
            img1_path: 第一张图像的路径
            img2_path: 第二张图像的路径
            detector_name: 特征检测器名称 ('SIFT', 'ORB', 'AKAZE', 'BRISK')
            matcher: 特征匹配器类型 ('BF' 或 'FLANN')
        """
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)
        
        if self.img1 is None or self.img2 is None:
            raise ValueError(f"无法读取图像: {img1_path} 或 {img2_path}")
            
        self.gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        
        # 创建特征检测器
        self.detector_name = detector_name
        self.detector = self._create_detector(detector_name)
        
        # 创建特征匹配器
        self.matcher_type = matcher
        self.matcher = self._create_matcher(matcher, detector_name)
        
        # 保存特征点和描述符
        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        self.matches = None
        self.y_diffs = None
        
    def _create_detector(self, detector_name):
        """创建特征检测器"""
        if detector_name == "SIFT":
            return cv2.SIFT.create(max_points=MAX_FEATURES)
        elif detector_name == "ORB":
            return cv2.ORB.create(nfeatures=MAX_FEATURES)
        elif detector_name == "AKAZE":
            return cv2.AKAZE.create(max_points=MAX_FEATURES)
        elif detector_name == "BRISK":
            logger.warning("BRISK detector has no max_points parameter")
            return cv2.BRISK.create()
        else:
            raise ValueError(f"不支持的特征检测器: {detector_name}")
            
    def _create_matcher(self, matcher_type, detector_name):
        """创建特征匹配器"""
        if matcher_type == "BF":
            # 根据检测器类型选择合适的距离度量
            if detector_name == "ORB" or detector_name == "BRISK":
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif matcher_type == "FLANN":
            # FLANN参数
            if detector_name == "ORB" or detector_name == "BRISK":
                # ORB和BRISK使用二进制描述符
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
            else:
                # SIFT和AKAZE使用浮点描述符
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"不支持的匹配器类型: {matcher_type}")
    
    def detect_and_compute(self):
        """检测特征点并计算描述符"""
        logger.info(f"使用 {self.detector_name} 检测特征...")
        self.kp1, self.des1 = self.detector.detectAndCompute(self.gray1, None)
        self.kp2, self.des2 = self.detector.detectAndCompute(self.gray2, None)
        
        logger.info(f"图像1检测到 {len(self.kp1)} 个特征点")
        logger.info(f"图像2检测到 {len(self.kp2)} 个特征点")
        
        return len(self.kp1), len(self.kp2)
    
    def match_features(self, ratio_test=0.75, min_match_count=10):
        """匹配两幅图像的特征点"""
        if self.des1 is None or self.des2 is None:
            self.detect_and_compute()
            
        # 确保有描述符可以匹配
        if self.des1 is None or self.des2 is None or len(self.des1) < 2 or len(self.des2) < 2:
            logger.error("没有足够的特征点用于匹配")
            return []
            
        logger.info(f"使用 {self.matcher_type} 匹配特征...")
        
        if self.matcher_type == "FLANN":
            # 对于FLANN匹配器，使用knnMatch
            matches = self.matcher.knnMatch(self.des1, self.des2, k=2)
            
            # 根据Lowe比率测试筛选好的匹配
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
            
            self.matches = good_matches
            
        else:  # BF匹配器
            self.matches = self.matcher.match(self.des1, self.des2)
            # 按距离排序
            self.matches = sorted(self.matches, key=lambda x: x.distance)
            # 只保留最好的匹配
            max_matches = min(len(self.matches), MAX_MATCHES)  # 限制最大匹配数量
            self.matches = self.matches[:max_matches]
            
        logger.info(f"找到 {len(self.matches)} 个好的匹配")
        
        return self.matches
    
    def calculate_y_misalignment(self):
        """计算y坐标错位"""
        if self.matches is None:
            self.match_features()
            
        if not self.matches:
            logger.error("没有匹配点，无法计算错位")
            return []
            
        # 计算每对匹配点的y坐标差异
        self.y_diffs = []
        for match in self.matches:
            # 获取匹配点的坐标
            pt1 = self.kp1[match.queryIdx].pt
            pt2 = self.kp2[match.trainIdx].pt
            
            # 计算y坐标差异
            y_diff = pt2[1] - pt1[1]
            self.y_diffs.append(y_diff)
            
        logger.info(f"计算了 {len(self.y_diffs)} 个匹配点的y坐标差异")
        
        return self.y_diffs
    
    def filter_matches_by_y_diff(self, c=0.10):
        """
        根据y坐标差异过滤匹配点
        
        参数:
            c: 标准差的倍数，用于定义接受范围 [mean - c*std, mean + c*std]
        
        返回:
            过滤后的匹配数量
        """
        if self.matches is None or self.y_diffs is None:
            self.calculate_y_misalignment()
            
        if not self.matches or not self.y_diffs:
            logger.error("没有匹配点可以过滤")
            return 0
            
        # 计算y坐标差异的均值和标准差
        mean_diff = np.mean(self.y_diffs)
        std_diff = np.std(self.y_diffs)

        from icecream import ic
        ic(mean_diff)
        ic(std_diff)
        ic(c)
        
        # 定义接受范围
        lower_bound = mean_diff - c * std_diff
        upper_bound = mean_diff + c * std_diff
        
        logger.info(f"过滤匹配点 - 均值: {mean_diff:.2f}, 标准差: {std_diff:.2f}")
        logger.info(f"接受范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # 过滤匹配点
        filtered_matches = []
        filtered_y_diffs = []
        
        for i, match in enumerate(self.matches):
            if lower_bound <= self.y_diffs[i] <= upper_bound:
                filtered_matches.append(match)
                filtered_y_diffs.append(self.y_diffs[i])
                
        # 更新匹配点和y坐标差异
        original_count = len(self.matches)
        self.matches = filtered_matches
        self.y_diffs = filtered_y_diffs
        
        filtered_count = len(self.matches)
        logger.info(f"过滤前: {original_count} 个匹配点, 过滤后: {filtered_count} 个匹配点")
        logger.info(f"移除了 {original_count - filtered_count} 个异常匹配点")
        
        return filtered_count
    
    def plot_y_misalignment_histogram(self, bins=50, save_path=None):
        """绘制y坐标错位直方图"""
        if self.y_diffs is None:
            self.calculate_y_misalignment()
            
        if not self.y_diffs:
            logger.error("没有y坐标差异数据可供绘图")
            return
            
        plt.figure(figsize=(10, 6))
        
        # 使用numpy计算直方图，以便找出最大计数的区间
        hist, bin_edges = np.histogram(self.y_diffs, bins=bins)
        
        # 找出最大计数的区间索引
        max_bin_idx = np.argmax(hist)
        
        # 计算该区间的中值
        max_bin_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
        
        # 绘制直方图
        plt.hist(self.y_diffs, bins=bins, color='blue', alpha=0.7)
        
        # 计算统计数据
        mean_diff = np.mean(self.y_diffs)
        median_diff = np.median(self.y_diffs)
        std_diff = np.std(self.y_diffs)
        
        # 添加垂直线表示均值、中位数和最大计数区间中值
        plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=2, label=f'均值: {mean_diff:.2f}')
        plt.axvline(median_diff, color='green', linestyle='dashed', linewidth=2, label=f'中位数: {median_diff:.2f}')
        plt.axvline(max_bin_center, color='orange', linestyle='dashed', linewidth=2, label=f'最大频率区间中值: {max_bin_center:.2f}')
        
        # 添加标题和标签
        plt.title(f'Y坐标错位统计 (标准差: {std_diff:.2f})')
        plt.xlabel('Y坐标差异（像素）')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加文本说明最大计数区间
        max_bin_text = f'最大频率区间: [{bin_edges[max_bin_idx]:.2f}, {bin_edges[max_bin_idx + 1]:.2f}], 计数: {hist[max_bin_idx]}'
        plt.figtext(0.5, 0.01, max_bin_text, ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"直方图已保存到: {save_path}")
            
        plt.show()
        
    def draw_matches(self, save_path=None):
        """绘制匹配结果"""
        if self.matches is None:
            self.match_features()
            
        if not self.matches:
            logger.error("没有匹配点，无法绘制")
            return
            
        # 绘制匹配
        img_matches = cv2.drawMatches(
            self.img1, self.kp1, 
            self.img2, self.kp2, 
            self.matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'特征匹配 ({len(self.matches)} 个匹配点)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"匹配图已保存到: {save_path}")
            
        plt.show()

    def get_filtered_median(self, c=0.010):
        """
        获取过滤后的y坐标差异中位数
        
        参数:
            c: 标准差的倍数，用于定义接受范围 [mean - c*std, mean + c*std]
        
        返回:
            float: 过滤后的y坐标差异中位数，如果没有匹配点则返回None
        """
        if self.matches is None or self.y_diffs is None:
            self.calculate_y_misalignment()
            
        if not self.matches or not self.y_diffs:
            logger.error("没有匹配点可以过滤")
            return None
            
        # 使用原有的过滤方法
        filtered_count = self.filter_matches_by_y_diff(c=c)
        
        if filtered_count == 0:
            logger.error("过滤后没有匹配点")
            return None
            
        # 计算并返回中位数
        median_diff = np.median(self.y_diffs)
        logger.info(f"过滤后的y坐标差异中位数: {median_diff:.2f}")
        
        return median_diff


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析图像对的y坐标错位')
    parser.add_argument('--img1', type=str, default='img0.png', help='第一张图像的路径')
    parser.add_argument('--img2', type=str, default='img1.png', help='第二张图像的路径')
    parser.add_argument('--detector', type=str, default='SIFT', 
                        choices=['SIFT', 'ORB', 'AKAZE', 'BRISK'], 
                        help='特征检测器类型')
    parser.add_argument('--matcher', type=str, default='BF', 
                        choices=['BF', 'FLANN'], 
                        help='特征匹配器类型')
    parser.add_argument('--output', type=str, default='results', 
                        help='输出文件夹路径')
    parser.add_argument('--filter', type=float, default=0.10,
                        help='过滤匹配点的标准差倍数 (默认: 0.10)')
    
    args = parser.parse_args()
    
    # 创建输出文件夹
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 创建分析器
    analyzer = ImagePairAnalyzer(args.img1, args.img2, args.detector, args.matcher)
    
    # 检测特征
    analyzer.detect_and_compute()
    
    analyzer.match_features()
    method_name = "std"
    
    # 计算y坐标错位
    analyzer.calculate_y_misalignment()
    
    # 绘制原始匹配结果
    raw_matches_path = output_dir / f"raw_matches_{method_name}_{args.detector}_{args.matcher}.png"
    analyzer.draw_matches(str(raw_matches_path))
    
    # 绘制原始y坐标错位直方图
    raw_histogram_path = output_dir / f"raw_y_misalignment_{method_name}_{args.detector}_{args.matcher}.png"
    analyzer.plot_y_misalignment_histogram(save_path=str(raw_histogram_path))
    
    # 过滤匹配点
    analyzer.filter_matches_by_y_diff(c=args.filter)

    
    # 绘制过滤后的匹配结果
    filtered_matches_path = output_dir / f"filtered_matches_{method_name}_{args.detector}_{args.matcher}.png"
    analyzer.draw_matches(str(filtered_matches_path))
    
    # 绘制过滤后的y坐标错位直方图
    filtered_histogram_path = output_dir / f"filtered_y_misalignment_{method_name}_{args.detector}_{args.matcher}.png"
    analyzer.plot_y_misalignment_histogram(save_path=str(filtered_histogram_path))
    
    logger.info("分析完成！")


if __name__ == "__main__":
    main() 