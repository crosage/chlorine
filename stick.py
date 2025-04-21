import re
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import rasterio
import sys

EXTENSION_TO_FORMAT = {
    '.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.tif': 'TIFF', '.tiff': 'TIFF'
}
FILENAME_PATTERN = re.compile(
    r"^(.*?)_x(\d+)_y(\d+).*?(\.jpg|\.jpeg|\.png|\.tif|\.tiff)$", re.IGNORECASE
)
OUTPUT_MIDDLE_PART = "_stitched"

# --- 辅助函数 ---

def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        identifier, row, col, ext = match.groups()
        if ext.lower() not in EXTENSION_TO_FORMAT:
            return None
        return identifier, int(row), int(col), ext.lower()
    return None

def get_required_tile_properties(filepath: Path, is_tiff: bool):
    width, height, channels, dtype, pillow_mode, profile = None, None, None, None, None, None
    try:
        if is_tiff:
            with rasterio.open(filepath) as src:
                width, height = src.width, src.height
                channels = src.count
                dtype = np.dtype(src.dtypes[0])
                profile = src.profile
                print(f"  基准属性 (TIFF) - 尺寸:{width}x{height}, 通道:{channels}, 类型:{dtype}")
        else:
            with Image.open(filepath) as img:
                width, height = img.size
                mode = img.mode
                if mode == 'L': channels = 1; dtype = np.uint8; pillow_mode = 'L'
                elif mode == 'RGB': channels = 3; dtype = np.uint8; pillow_mode = 'RGB'
                elif mode == 'RGBA': channels = 4; dtype = np.uint8; pillow_mode = 'RGBA'
                else:
                    print(f"  警告: 基准切片模式为 '{mode}'。将尝试转换为 RGB。如果后续切片不匹配或转换失败，可能出错。")
                    try:
                        img_conv = img.convert('RGB')
                        pillow_mode = 'RGB'; channels = 3; dtype = np.uint8
                    except Exception as e:
                        raise ValueError(f"无法将模式 '{mode}' 转换为 RGB: {e}")

                if pillow_mode is None: raise ValueError("未能确定有效的 Pillow 模式")
                print(f"  基准属性 (Pillow) - 尺寸:{width}x{height}, 模式:{pillow_mode}, 通道:{channels}, 类型:{dtype}")

        if not all([width is not None, height is not None, channels is not None, dtype is not None]):
             raise ValueError("未能获取完整的基准属性")

        return width, height, channels, dtype, pillow_mode, profile

    except (FileNotFoundError, rasterio.RasterioIOError, Image.UnidentifiedImageError, ValueError, Exception) as e:
        print(f"错误: 读取基准切片 '{filepath}' 失败: {e}", file=sys.stderr)
        raise RuntimeError(f"无法处理基准切片 {filepath.name}") from e


def save_stitched_image(canvas: np.ndarray, output_filepath: Path, is_tiff: bool,
                        base_profile: dict, base_pillow_mode: str,
                        total_height: int, total_width: int, channels: int, dtype: np.dtype,
                        origin_transform=None, origin_crs=None):
    print(f"  正在保存: {output_filepath}")
    try:
        if is_tiff:
            if not base_profile: raise ValueError("缺少 TIFF Profile 信息")
            profile = base_profile.copy()
            profile.update({
                'height': total_height, 'width': total_width, 'count': channels,
                'dtype': dtype, 'driver': 'GTiff',
                'compress': profile.get('compress', 'deflate'),
                'tiled': profile.get('tiled', True)
            })
            if profile['tiled'] and not ('blockxsize' in profile and 'blockysize' in profile):
                block_size = 256
                profile['blockxsize'] = min(block_size, profile['width'])
                profile['blockysize'] = min(block_size, profile['height'])

            if origin_transform:
                profile['transform'] = origin_transform
                profile['crs'] = origin_crs
            else:
                 print("    警告: 未找到原点 (0,0) 的有效 Transform。可能无地理参考或使用基准切片的参考。")
                 if 'transform' not in base_profile: profile.pop('transform', None)
                 if 'crs' not in base_profile: profile.pop('crs', None)

            profile.pop('nodata', None)

            with rasterio.open(output_filepath, 'w', **profile) as dest:
                dest.write(canvas.transpose(2, 0, 1) if channels > 1 else canvas,
                           indexes=list(range(1, channels + 1)) if channels > 1 else 1)
            print(f"    Rasterio 保存 TIFF 成功!")

        else:
            if not base_pillow_mode: raise ValueError("缺少 Pillow 模式信息")
            try:
                stitched_image = Image.fromarray(canvas, mode=base_pillow_mode)
            except Exception as e:
                raise ValueError(f"从 NumPy 数组创建 Pillow 图像失败 (模式:{base_pillow_mode}, 类型:{canvas.dtype}): {e}")

            save_options = {}
            img_to_save = stitched_image
            output_format_string = EXTENSION_TO_FORMAT[output_filepath.suffix.lower()]

            if output_format_string == 'JPEG':
                save_options['quality'] = 95
                if img_to_save.mode != 'RGB':
                    print(f"    注意: 转换为 RGB 模式以保存为 JPEG。")
                    try:
                        img_to_save = img_to_save.convert('RGB')
                    except Exception as conv_e:
                        raise ValueError(f"无法转换为 RGB 以保存 JPEG: {conv_e}")
            elif output_format_string == 'PNG':
                save_options['compress_level'] = 6

            img_to_save.save(output_filepath, format=output_format_string, **save_options)
            print(f"    Pillow 保存 {output_format_string} 成功!")

    except (rasterio.RasterioIOError, ValueError, TypeError, Exception) as e:
        print(f"错误: 保存文件 '{output_filepath}' 失败: {e}", file=sys.stderr)
        raise RuntimeError("保存失败") from e
def stitch_images(input_dir: Path, output_dir: Path):
    if not input_dir.is_dir():
        print(f"错误: 输入路径 '{input_dir}' 无效。", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 扫描并分组
    tiles_by_identifier = defaultdict(list)
    print(f"正在扫描文件夹: {input_dir}")
    for filepath in input_dir.iterdir():
        if filepath.is_file():
            parsed = parse_filename(filepath.name)
            if parsed:
                tiles_by_identifier[parsed[0]].append({
                    "path": filepath, "row": parsed[1], "col": parsed[2], "ext": parsed[3]
                })

    if not tiles_by_identifier:
        print("错误: 未找到符合 '名称_x行号_y列号.扩展名' 规则的文件。", file=sys.stderr)
        return
    print(f"找到 {len(tiles_by_identifier)} 个标识符。")

    # 2. 逐个标识符处理
    for identifier, tiles in tiles_by_identifier.items():
        print(f"\n--- 处理标识符: {identifier} ---")
        if not tiles: continue

        first_tile_info = tiles[0]
        output_ext = first_tile_info['ext']
        is_tiff = output_ext in ['.tif', '.tiff']

        try:
            # 2a. 获取基准属性 (失败则跳过此标识符)
            tile_width, tile_height, channels, dtype, pillow_mode, profile = get_required_tile_properties(first_tile_info['path'], is_tiff)
            # 2b. 计算网格和画布尺寸
            max_row = max(t['row'] for t in tiles)
            max_col = max(t['col'] for t in tiles)
            grid_rows, grid_cols = max_row + 1, max_col + 1
            total_height = grid_rows * tile_height
            total_width = grid_cols * tile_width
            print(f"  网格: {grid_rows}x{grid_cols}, 总尺寸: {total_width}x{total_height}")
            # 2c. 创建画布
            canvas_shape = (total_height, total_width, channels) if channels > 1 else (total_height, total_width)
            if total_height <= 0 or total_width <= 0: raise ValueError("画布尺寸无效")
            canvas = np.zeros(canvas_shape, dtype=dtype)
            print(f"  已创建画布, 形状: {canvas.shape}, 类型: {canvas.dtype}")
            # 2d. 粘贴切片
            print("  正在粘贴切片 (确保后续文件属性一致)")
            processed_count = 0
            for tile_info in tiles:
                tile_path = tile_info['path']
                try:
                    tile_array = None
                    if is_tiff:
                        with rasterio.open(tile_path) as src:
                            if src.width != tile_width or src.height != tile_height:
                                print(f"  警告: 切片 {tile_path.name} 尺寸与基准不符! 跳过。")
                                continue
                            data = src.read()
                            tile_array = data.transpose(1, 2, 0) if channels > 1 else data[0]
                    else:
                        with Image.open(tile_path) as img:
                            if img.width != tile_width or img.height != tile_height:
                                print(f"  警告: 切片 {tile_path.name} 尺寸与基准不符! 跳过。")
                                continue
                            img_to_process = img
                            if img.mode != pillow_mode:
                                try:
                                    img_to_process = img.convert(pillow_mode)
                                except Exception:
                                     print(f"  警告: 尝试转换切片 {tile_path.name} 到模式 {pillow_mode} 失败。跳过。")
                                     continue
                            tile_array = np.array(img_to_process)
                    x_pos = tile_info['col'] * tile_width
                    y_pos = tile_info['row'] * tile_height
                    if y_pos + tile_height > total_height or x_pos + tile_width > total_width:
                         print(f"  警告: 计算出的切片 {tile_path.name} 位置超出画布边界。跳过。")
                         continue

                    if channels > 1:
                        canvas[y_pos : y_pos + tile_height, x_pos : x_pos + tile_width, :] = tile_array
                    else:
                        canvas[y_pos : y_pos + tile_height, x_pos : x_pos + tile_width] = tile_array
                    processed_count += 1

                except (FileNotFoundError, rasterio.RasterioIOError, Image.UnidentifiedImageError, Exception) as e:
                    print(f"错误: 处理切片 '{tile_path}' 时出错: {e}。跳过此切片。", file=sys.stderr)

            print(f"  完成粘贴尝试，共处理 {processed_count} / {len(tiles)} 个切片。")

            # 2e. 保存结果 (如果至少粘贴了一个)
            if processed_count > 0:
                 # 获取原点(0,0)的地理参考信息 (仍然需要)
                 origin_transform, origin_crs = None, profile.get('crs') if profile else None
                 for t in tiles:
                     if t['row'] == 0 and t['col'] == 0:
                         try:
                             with rasterio.open(t['path']) as origin_src:
                                 if origin_src.width == tile_width and origin_src.height == tile_height:
                                     origin_transform = origin_src.transform
                                     origin_crs = origin_src.crs
                         except Exception: pass
                         break

                 output_filename = f"{identifier}{OUTPUT_MIDDLE_PART}{output_ext}"
                 output_filepath = output_dir / output_filename
                 save_stitched_image(canvas, output_filepath, is_tiff, profile, pillow_mode,
                                     total_height, total_width, channels, dtype,
                                     origin_transform, origin_crs)
            else:
                print("  没有成功处理任何切片，不保存此标识符的结果。")

        except (RuntimeError, MemoryError, ValueError, TypeError) as e:
             print(f"错误: 处理标识符 '{identifier}' 失败 ({e})。跳过。", file=sys.stderr)
             continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "拼接遵循 '名称_x行号_y列号.扩展名' 规则的图像切片 (简化版)。\n"
            "**重要:** 假设同一标识符下的所有切片属性完全一致。\n"
            "行号(x)从0开始，列号(y)从0开始。"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_dir", type=str, help="包含图像切片的文件夹路径。")
    parser.add_argument("output_dir", type=str, help="保存拼接后图像的文件夹路径。")
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    if not input_path.is_dir():
         print(f"错误: 输入路径 '{input_path}' 无效。", file=sys.stderr)
         sys.exit(1)

    print(f"输入文件夹: {input_path}")
    print(f"输出文件夹: {output_path}")
    print("--- 开始处理 (简化模式) ---")
    print("**注意: 脚本假设同一标识符下的切片属性一致，仅验证第一个切片。**")

    stitch_images(input_path, output_path)

    print("\n脚本执行完毕。")