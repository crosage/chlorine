from PIL import Image
from collections import Counter
import sys
import os
Image.MAX_IMAGE_PIXELS = None
def count_pixel_colors(image_path):
    if not os.path.exists(image_path):
        print(f"错误：文件不存在 '{image_path}'")
        return None

    try:
        with Image.open(image_path) as img:
            print(f"正在打开图像: {image_path}")
            print(f"图像模式: {img.mode}, 尺寸: {img.size}")
            img_rgb = img.convert('RGB')
            print(f"已转换为 RGB 模式进行处理。")
            pixel_data = img_rgb.getdata()
            print("正在统计像素颜色...")
            rgb_counts = Counter(pixel_data)
            print("统计完成。")

            return rgb_counts

    except FileNotFoundError:
        print(f"错误：文件未找到 '{image_path}' (Pillow 错误)")
        return None
    except Image.UnidentifiedImageError:
        print(f"错误：无法识别的图像文件格式或文件已损坏 '{image_path}'")
        return None
    except Exception as e:
        print(f"处理图像时发生未知错误：{e}")
        return None
if __name__ == "__main__":
    image_file = input("请输入图片文件的完整路径: ")
    color_counts = count_pixel_colors(image_file)
    if color_counts:
        total_pixels = sum(color_counts.values())
        unique_colors = len(color_counts)

        print(f"\n--- 统计结果 ('{os.path.basename(image_file)}') ---")
        print(f"总像素数: {total_pixels}")
        print(f"独立 RGB 颜色数量: {unique_colors}")
        sorted_counts = color_counts.most_common()
        print("\n各 RGB 颜色及其出现次数 (按次数降序):")
        limit = 20
        for i, (rgb, count) in enumerate(sorted_counts):
            if i >= limit:
                print(f"\n... (还有 {unique_colors - limit} 种其他颜色未显示)")
                break
            print(f"  排名 {i+1}: RGB={rgb}, 次数={count}")

    else:
        print("\n未能完成像素统计。请检查文件路径和文件是否有效。")