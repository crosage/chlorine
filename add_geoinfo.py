import os
import rasterio
import numpy as np
import typer
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import traceback
Image.MAX_IMAGE_PIXELS = None
# 创建一个 Typer 应用实例
app = typer.Typer(
    name="geotiff-mask-processor",
    help="一个将普通图像掩码（如PNG）根据源栅格文件赋予地理信息的CLI工具。",
    add_completion=False
)

def match_files_generic(source_dir: Path, label_dir: Path) -> list:

    print(f"INFO: 开始进行宽泛文件匹配 (忽略后缀) | 源: '{source_dir.name}', 标签: '{label_dir.name}'")
    label_map = {f.stem: f for f in label_dir.iterdir() if f.is_file()}

    if not label_map:
        print(f"WARNING: 在标签文件夹 '{label_dir}' 中没有找到任何文件。")
        return []

    matched_pairs = []
    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        source_stem = source_file.stem
        if source_stem in label_map:
            matched_pairs.append(
                (str(source_file), str(label_map[source_stem]))
            )

    print(f"INFO: 成功匹配到 {len(matched_pairs)} 对文件。")
    return matched_pairs

def add_geoinfo_to_mask(source_geofile_path: str, mask_image_path: str, output_geotiff_path: str):

    try:
        # 使用Pillow打开掩码图像并转换为Numpy数组
        mask_image = Image.open(mask_image_path)
        mask_data = np.array(mask_image)

        # 从源GeoTIFF读取地理空间元数据
        with rasterio.open(source_geofile_path) as src:
            profile = src.profile
            # 检查源文件和掩码文件的尺寸是否一致
            if src.height != mask_data.shape[0] or src.width != mask_data.shape[1]:
                tqdm.write(
                    f"WARNING: 文件 '{Path(source_geofile_path).name}' 和 '{Path(mask_image_path).name}' 的尺寸不匹配！")
                tqdm.write(f"  - 源文件尺寸 (高x宽): {src.height} x {src.width}")
                tqdm.write(f"  - 掩码尺寸 (高x宽): {mask_data.shape[0]} x {mask_data.shape[1]}")

        # 判断掩码是单通道（灰度图）还是多通道（彩色图）
        if len(mask_data.shape) == 2:
            num_bands = 1
            # 为单通道数据增加一个维度以符合rasterio的写入要求
            mask_data_reshaped = np.expand_dims(mask_data, axis=0)
        else:
            num_bands = mask_data.shape[2]
            # 调整多通道数据的维度顺序 (H, W, C) -> (C, H, W)
            mask_data_reshaped = np.moveaxis(mask_data, -1, 0)

        # 更新profile以匹配掩码数据的属性
        profile.update({
            'driver': 'GTiff',
            'dtype': mask_data.dtype,
            'count': num_bands,
            'compress': 'lzw'  # 使用lzw无损压缩
        })
        # 移除可能不适用的nodata值
        profile.pop('nodata', None)

        # 写入新的GeoTIFF文件
        with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
            dst.write(mask_data_reshaped)

    except Exception as e:
        # 捕获并打印处理过程中的任何错误
        tqdm.write(f"\nERROR: 处理文件 '{Path(mask_image_path).name}' 时发生错误: {e}")
        tqdm.write(traceback.format_exc())


@app.command()
def process_geotiffs(
    source_folder: Path = typer.Option(
        ...,
        '--source', '-s',
        help="包含地理信息的源图像文件夹路径 (例如 .tif, .vrt)。",
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True
    ),
    label_folder: Path = typer.Option(
        ...,
        '--label', '-l',
        help="包含掩码标签的图像文件夹路径 (例如 .png, .jpg)。",
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True
    ),
    output_folder: Path = typer.Option(
        ...,
        '--output', '-o',
        help="用于保存带有地理信息的GeoTIFF掩码的输出文件夹路径。",
        file_okay=False, dir_okay=True, writable=True, resolve_path=True
    )
):
    """
    自动化处理流程：匹配源文件和标签，并将地理信息赋予标签，生成新的GeoTIFF掩码。
    """
    # 支持的文件扩展名列表
    VALID_SOURCE_EXTENSIONS = ['.tif', '.tiff', '.vrt', '.img']
    VALID_MASK_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif']

    print("自动化处理流程启动...")
    # 确保输出文件夹存在
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"输出文件夹: {output_folder}")

    # 步骤 1: 匹配文件
    matched_pairs = match_files_generic(source_folder, label_folder)

    if not matched_pairs:
        print("\n未找到任何可匹配的文件。程序退出。")
        raise typer.Exit()

    # 步骤 2: 验证并处理匹配的文件对
    print("\n开始验证并处理匹配的文件对...")
    processed_count = 0
    for source_path_str, label_path_str in tqdm(matched_pairs, desc="处理文件"):
        source_path = Path(source_path_str)
        label_path = Path(label_path_str)

        # 验证文件后缀名是否有效
        is_source_valid = source_path.suffix.lower() in VALID_SOURCE_EXTENSIONS
        is_label_valid = label_path.suffix.lower() in VALID_MASK_EXTENSIONS

        if not is_source_valid or not is_label_valid:
            tqdm.write(f"SKIPPING: 文件对 '{source_path.name}' 和 '{label_path.name}' 的后缀名不受支持。")
            continue

        # 构建输出文件路径
        base_filename = label_path.stem
        output_path = output_folder / f"{base_filename}.tif"

        # 核心处理函数
        add_geoinfo_to_mask(str(source_path), str(label_path), str(output_path))
        processed_count += 1

    print(f"\n处理完成！共处理了 {processed_count} / {len(matched_pairs)} 对有效文件。")
    print(f"所有新的GeoTIFF掩码已保存到: {output_folder}")


if __name__ == "__main__":
    app()
