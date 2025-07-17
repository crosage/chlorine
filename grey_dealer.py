import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def process_images_inplace(input_folder, value_map):
    input_path = Path(input_folder)
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

    if not input_path.is_dir():
        print(f"错误：输入文件夹 '{input_folder}' 不存在或不是一个目录。")
        return

    print("="*60)
    print("警告：此脚本将直接修改并覆盖输入文件夹中的原始图像文件！")
    print(f"目标文件夹: {input_path.resolve()}")
    print("强烈建议在继续操作前备份您的数据。")
    print("="*60)
    try:
        for i in range(5, 0, -1):
            print(f"操作将在 {i} 秒后开始... (按 Ctrl+C 取消)")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n操作已取消。")
        return

    print("开始处理...")

    processed_count = 0
    skipped_count = 0

    for item in input_path.iterdir():
        if item.is_file() and item.suffix.lower() in supported_extensions:
            try:
                input_file = str(item.resolve())
                image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"警告：无法读取图像文件 '{item.name}'，跳过。")
                    skipped_count += 1
                    continue

                modified_image = image.copy()

                applied = False
                for src_val, tgt_val in value_map.items():
                    mask = (modified_image == src_val)
                    if np.any(mask):
                        modified_image[mask] = tgt_val
                        applied = True

                if applied:
                    cv2.imwrite(input_file, modified_image)
                    print(f"已修改并覆盖 '{item.name}'")
                    processed_count += 1
                else:
                    print(f"文件 '{item.name}' 未包含待替换的灰度值，跳过。")
                    skipped_count += 1

            except Exception as e:
                print(f"处理文件 '{item.name}' 时发生错误: {e}")
                skipped_count += 1
        elif item.is_file():
            skipped_count += 1

    print("\n处理完成。")
    print(f"成功修改文件数: {processed_count}")
    print(f"跳过/失败文件数: {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将图像中指定灰度值替换为新的值，并直接覆盖原文件。")
    parser.add_argument("input_folder", help="包含要就地修改图像的文件夹路径。")
    parser.add_argument(
        "--map", nargs="+", required=True,
        help="灰度值替换映射，例如 255:1 128:0 表示将255变为1，128变为0"
    )

    args = parser.parse_args()

    # 将 '255:1' 这类字符串转为字典
    try:
        value_map = {int(pair.split(":")[0]): int(pair.split(":")[1]) for pair in args.map}
    except Exception as e:
        print(f"错误：无法解析灰度映射: {e}")
        exit(1)

    process_images_inplace(args.input_folder, value_map)
