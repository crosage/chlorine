import os
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import rasterio
from rasterio.enums import Resampling
import sys # Import sys for sys.exit

EXTENSION_TO_FORMAT = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.png': 'PNG',
    '.tif': 'TIFF',
    '.tiff': 'TIFF'
}

FILENAME_PATTERN = re.compile(
    r"^(.*?)_x(\d+)_y(\d+).*?(\.jpg|\.jpeg|\.png|\.tif|\.tiff)$",
    re.IGNORECASE
)
OUTPUT_MIDDLE_PART = "_stitched"

def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        identifier = match.group(1)
        x = int(match.group(2))
        y = int(match.group(3))
        extension = match.group(4).lower()
        return identifier, x, y, extension
    else:
        return None
def stitch_images(input_dir: Path, output_dir: Path):
    if not input_dir.is_dir():
        print(f"错误: 输入路径 '{input_dir}' 不是一个有效的文件夹。")
        sys.exit(1) # Exit if input dir is invalid

    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_by_identifier = defaultdict(list)
    print(f"正在扫描文件夹: {input_dir}")
    for filepath in input_dir.iterdir():
        if filepath.is_file():
            parsed_info = parse_filename(filepath.name)
            if parsed_info:
                identifier, x, y, extension = parsed_info
                tiles_by_identifier[identifier].append({
                    "path": filepath, "x": x, "y": y, "ext": extension
                })

    if not tiles_by_identifier:
        print("在输入文件夹中没有找到符合命名规则的文件。")
        print(f"文件名需要匹配: (名称)_x(数字)_y(数字).(jpg/jpeg/png/tif/tiff)")
        return # No need to exit, just inform and finish

    print(f"找到了 {len(tiles_by_identifier)} 个不同的图像标识符进行拼接。")

    for identifier, original_tiles in tiles_by_identifier.items():
        print(f"\n正在处理标识符: {identifier}")
        if not original_tiles: continue
        tiles = original_tiles

        if not tiles:
             print(f" 标识符 '{identifier}' 没有找到有效的切片。跳过。")
             continue

        first_tile_info = tiles[0]
        output_ext = first_tile_info['ext']
        is_output_tiff = output_ext in ['.tif', '.tiff']
        output_format_string = EXTENSION_TO_FORMAT.get(output_ext) # Pillow 格式

        if not output_format_string:
            print(f" 错误: 不支持的文件扩展名 '{output_ext}'。跳过。")
            continue

        print(f" 检测到输入格式为 {output_ext}。")
        if is_output_tiff:
            print(" 将使用 Rasterio 保存输出 TIFF (尝试保留地理信息)。")
        else:
            print(f" 将使用 Pillow 保存输出 {output_format_string}。")


        tile_width, tile_height, channels, determined_dtype = None, None, None, None
        first_tile_profile = None
        pillow_mode = None

        try:
            # --- 读取第一个切片信息 (use the potentially filtered list) ---
            if is_output_tiff:
                print(f" 使用 Rasterio 读取第一个 TIFF 切片信息: {first_tile_info['path'].name}")
                with rasterio.open(first_tile_info['path']) as src:
                    tile_width = src.width
                    tile_height = src.height
                    channels = src.count
                    determined_dtype = np.dtype(src.dtypes[0])
                    first_tile_profile = src.profile
            else:
                print(f" 使用 Pillow 读取第一个切片信息: {first_tile_info['path'].name}")
                with Image.open(first_tile_info['path']) as img:
                    tile_width, tile_height = img.size
                    mode = img.mode
                    if mode == 'L': channels = 1; determined_dtype = np.uint8
                    elif mode == 'RGB': channels = 3; determined_dtype = np.uint8
                    elif mode == 'RGBA': channels = 4; determined_dtype = np.uint8
                    elif mode == 'P':
                        try: # Add try-except for conversion just in case
                            img_conv = img.convert('RGB'); mode = img_conv.mode
                            channels = 3; determined_dtype = np.uint8
                        except Exception as conv_e:
                             print(f"   警告: 转换 P 模式图像 '{first_tile_info['path'].name}' 到 RGB 失败: {conv_e}。 尝试RGBA。")
                             try:
                                 img_conv = img.convert('RGBA'); mode = img_conv.mode
                                 channels = 4; determined_dtype = np.uint8
                             except Exception as conv_e2:
                                  print(f"   错误: 转换 P 模式图像 '{first_tile_info['path'].name}' 到 RGBA 也失败: {conv_e2}。跳过。")
                                  continue # Skip this identifier if conversion fails

                    elif mode == 'I;16': channels = 1; determined_dtype = np.uint16
                    elif mode == 'I': channels = 1; determined_dtype = np.int32
                    elif mode == 'F': channels = 1; determined_dtype = np.float32
                    else: # Default conversion fallback
                        print(f"   警告: 未知模式 '{mode}' for '{first_tile_info['path'].name}'. 尝试转换为 RGB。")
                        try:
                             img_conv = img.convert('RGB'); mode = img_conv.mode
                             channels = 3; determined_dtype = np.uint8
                        except Exception as conv_e:
                            print(f"   错误: 转换未知模式图像 '{first_tile_info['path'].name}' 到 RGB 失败: {conv_e}。跳过。")
                            continue # Skip this identifier

                    pillow_mode = mode # Save the final mode used

        except Exception as e:
            print(f"错误: 无法读取第一个切片文件 '{first_tile_info['path']}' 来确定属性: {e}")
            continue

        if not all([tile_width is not None, tile_height is not None, channels is not None, determined_dtype is not None]):
            print(f"错误: 未能成功确定第一个切片的完整属性。跳过 '{identifier}'.")
            continue

        print(f" 切片尺寸: {tile_width}x{tile_height}, 通道: {channels}, 类型: {determined_dtype}")

        # --- Calculations based on the potentially filtered 'tiles' list ---
        max_x = max(tile['x'] for tile in tiles)
        max_y = max(tile['y'] for tile in tiles) # This max_y is now correct based on filtered tiles
        grid_width = max_x + 1
        grid_height = max_y + 1 # This grid_height is now correct
        total_width = grid_width * tile_width
        total_height = grid_height * tile_height # This total_height uses the potentially reduced grid_height

        print(f" 计算网格大小: {grid_width}x{grid_height} (基于找到/过滤后的切片)")
        print(f" 最终图像尺寸: {total_width}x{total_height}")

        # 创建 NumPy 画布
        canvas_shape = (total_height, total_width, channels) if channels > 1 else (total_height, total_width)
        try:
            # Check for zero dimensions before creating canvas
            if total_height <= 0 or total_width <= 0:
                 print(f"错误: 计算得到的画布尺寸无效 ({total_height}x{total_width})。跳过 '{identifier}'。")
                 continue
            canvas = np.zeros(canvas_shape, dtype=determined_dtype)
            print(f" 已创建画布，形状: {canvas.shape}, 类型: {canvas.dtype}")
        except MemoryError:
            print(f"错误: 创建画布时内存不足！尺寸: {canvas_shape}, 类型: {determined_dtype}。尝试减小 --max-rows (如果适用)。")
            continue
        except ValueError as ve:
             print(f"错误: 创建画布时值错误 (可能是类型或形状问题): {ve}")
             continue

        print(" 正在粘贴切片...")
        processed_tiles = 0
        # Iterate through the potentially filtered 'tiles' list
        for tile_info in tiles:
            try:
                tile_array = None
                if tile_info['ext'] in ['.tif', '.tiff']:
                    with rasterio.open(tile_info['path']) as tile_src:
                        # Check consistency against the *first* tile's properties
                        if tile_src.width != tile_width or tile_src.height != tile_height \
                           or tile_src.count != channels or np.dtype(tile_src.dtypes[0]) != determined_dtype:
                            print(f"  警告: 切片 {tile_info['path'].name} 属性与第一个切片不符，跳过。")
                            continue
                        data = tile_src.read() # C, H, W
                        if channels > 1: tile_array = data.transpose(1, 2, 0) # H, W, C
                        else: tile_array = data[0] # H, W
                else: # 使用 Pillow 读取
                    with Image.open(tile_info['path']) as tile_img:
                        if tile_img.width != tile_width or tile_img.height != tile_height:
                            print(f"  警告: 切片 {tile_info['path'].name} 尺寸不符，跳过。")
                            continue
                        # Mode consistency check/conversion (using determined pillow_mode)
                        current_mode = tile_img.mode
                        final_tile_img = tile_img
                        if pillow_mode and current_mode != pillow_mode:
                           # Handle 'P' mode specifically if target is RGB/RGBA
                           if current_mode == 'P' and pillow_mode in ['RGB', 'RGBA']:
                               try:
                                   final_tile_img = tile_img.convert(pillow_mode)
                               except Exception as conv_e:
                                   print(f"  警告: 转换切片 {tile_info['path'].name} (模式 {current_mode}) 到目标模式 {pillow_mode} 失败: {conv_e}。跳过。")
                                   continue
                           # Generic conversion attempt only if modes differ significantly
                           elif current_mode != pillow_mode: # Avoid unnecessary conversions if e.g. both are RGB
                                print(f"  注意: 切片 {tile_info['path'].name} (模式 {current_mode}) 与目标模式 {pillow_mode} 不同。尝试转换。")
                                try:
                                   final_tile_img = tile_img.convert(pillow_mode)
                                except Exception as conv_e:
                                   print(f"  警告: 转换切片 {tile_info['path'].name} 到目标模式 {pillow_mode} 失败: {conv_e}。跳过。")
                                   continue

                        tile_array = np.array(final_tile_img)
                        # Type check/conversion (ensure consistency)
                        if tile_array.dtype != determined_dtype:
                             try:
                                 # Be cautious with type conversions, ensure they make sense
                                 # e.g., float to int might lose data
                                 print(f"  注意: 切片 {tile_info['path'].name} 类型 ({tile_array.dtype}) 与目标类型 ({determined_dtype}) 不同。尝试转换。")
                                 tile_array = tile_array.astype(determined_dtype)
                             except Exception as dtype_e:
                                 print(f"  错误: 转换切片 {tile_info['path'].name} 类型失败: {dtype_e}。跳过。")
                                 continue


                # --- Pasting logic remains the same ---
                if tile_array is not None:
                    print(f"名字：{tile_info['path'].name}   {tile_info['x']}    {tile_info['y']}")
                    # Calculate position based on tile's y coordinate (which is already < max_rows if filtering applied)
                    x_pos = tile_info['x'] * tile_width
                    y_pos = tile_info['y'] * tile_height

                    # Boundary check (shouldn't be necessary if canvas size is correct, but safe)
                    if y_pos + tile_height > total_height or x_pos + tile_width > total_width:
                        print(f"   警告: 切片 {tile_info['path'].name} 坐标 ({tile_info['x']},{tile_info['y']}) 超出计算的画布边界。跳过。")
                        continue

                    if channels > 1:
                        # Ensure tile_array has the correct number of channels
                        if tile_array.shape[2] != channels:
                             print(f"   警告: 切片 {tile_info['path'].name} 通道数 ({tile_array.shape[2]}) 与预期 ({channels}) 不符。跳过。")
                             continue
                        canvas[y_pos : y_pos + tile_height, x_pos : x_pos + tile_width, :] = tile_array
                    else:
                        # Ensure tile_array is 2D for single channel
                        if tile_array.ndim != 2:
                             print(f"   警告: 单通道切片 {tile_info['path'].name} 维度 ({tile_array.ndim}) 不正确。跳过。")
                             continue
                        canvas[y_pos : y_pos + tile_height, x_pos : x_pos + tile_width] = tile_array
                    processed_tiles += 1

            except FileNotFoundError:
                 print(f"错误: 找不到切片文件 '{tile_info['path']}'。")
            except rasterio.RasterioIOError as rio_e:
                 print(f"错误: Rasterio 读取切片 '{tile_info['path']}' 时出错: {rio_e}")
            except Image.UnidentifiedImageError:
                 print(f"错误: Pillow 无法识别或打开切片 '{tile_info['path']}'。")
            except Exception as e:
                 print(f"错误: 处理/粘贴切片 '{tile_info['path']}' 时发生意外错误: {e}")

        print(f"粘贴完成 {processed_tiles}/{len(tiles)} 个切片。")

        # --- Saving logic remains largely the same ---
        if processed_tiles > 0:
            output_filename = f"{identifier}{OUTPUT_MIDDLE_PART}{output_ext}"
            output_filepath = output_dir / output_filename

            # --- Use Rasterio for TIFF ---
            if is_output_tiff and first_tile_profile is not None:
                try:
                    print(f"正在准备使用 Rasterio 保存 TIFF: {output_filepath}")
                    stitched_profile = first_tile_profile.copy()
                    # Use calculated total_height and total_width which respect max_rows
                    stitched_profile['height'] = total_height
                    stitched_profile['width'] = total_width
                    stitched_profile['count'] = channels
                    stitched_profile['dtype'] = determined_dtype

                    # Find the tile corresponding to x=0, y=0 (if it exists in the filtered set)
                    # This gives a more accurate starting transform.
                    origin_tile_path = None
                    origin_transform = None
                    for t in tiles:
                        if t['x'] == 0 and t['y'] == 0:
                             origin_tile_path = t['path']
                             break

                    if origin_tile_path:
                         try:
                             with rasterio.open(origin_tile_path) as origin_src:
                                 origin_transform = origin_src.transform
                                 print(f"使用切片 '{origin_tile_path.name}' (x=0, y=0) 的 Transform。")
                         except Exception as origin_e:
                             print(f"警告: 无法读取 x=0, y=0 切片 '{origin_tile_path.name}' 的 Transform: {origin_e}。将使用第一个切片的 Transform。")
                             origin_transform = first_tile_profile.get('transform') # Fallback to first tile's
                    else:
                         print(f"警告: 在 (过滤后的) 切片中未找到 x=0, y=0 的切片。将使用第一个切片的 Transform 作为地理参考起点。")
                         origin_transform = first_tile_profile.get('transform') # Fallback to first tile's


                    if origin_transform:
                         stitched_profile['transform'] = origin_transform
                    else:
                         # If no transform anywhere, remove it to avoid errors
                         stitched_profile.pop('transform', None)
                         stitched_profile.pop('crs', None) # Also remove CRS if transform is missing
                         print("警告: 未能找到有效的地理参考信息 (Transform)，输出 TIFF 将不包含地理参考。")


                    stitched_profile['driver'] = 'GTiff'
                    # Sensible compression defaults
                    if 'compress' not in stitched_profile or stitched_profile['compress'] is None:
                           stitched_profile['compress'] = 'deflate'
                    if 'tiled' not in stitched_profile or not stitched_profile['tiled']:
                           stitched_profile['tiled'] = True
                           stitched_profile['blockxsize'] = 256 if tile_width >= 256 else tile_width
                           stitched_profile['blockysize'] = 256 if tile_height >= 256 else tile_height
                           # Ensure block sizes are powers of 2 or multiples of 16 if possible for better compatibility
                           # This is simplified, a more robust check might be needed

                    # Remove keys that might cause issues if copied directly and not updated
                    stitched_profile.pop('nodata', None) # Nodata might vary, safer to remove unless specifically handled

                    with rasterio.open(output_filepath, 'w', **stitched_profile) as dest:
                        if channels == 1:
                            print(f"写入单波段数据...")
                            dest.write(canvas, 1)
                        elif channels > 1:
                            print(f"转换数据到 (C, H, W) 顺序并写入 {channels} 波段...")
                            data_to_write = canvas.transpose(2, 0, 1)
                            dest.write(data_to_write)
                        print(f"Rasterio 保存 TIFF 成功!")

                except Exception as e:
                    print(f"错误: 使用 Rasterio 保存 TIFF 文件 '{output_filepath}' 失败: {e}")
                    print(f"尝试回退到 Pillow 保存 (无地理信息)...")
                    # --- Fallback to Pillow for TIFF ---
                    try:
                        # Attempt to determine fallback pillow mode if not already set
                        if pillow_mode is None:
                            if channels==1: pillow_mode = 'L' if determined_dtype == np.uint8 else ('I;16' if determined_dtype==np.uint16 else ('I' if determined_dtype==np.int32 else ('F' if determined_dtype==np.float32 else None)))
                            elif channels==3: pillow_mode = 'RGB'
                            elif channels==4: pillow_mode = 'RGBA'
                            else: pillow_mode = None # Cannot determine fallback

                        if pillow_mode:
                            stitched_image = Image.fromarray(canvas, mode=pillow_mode)
                            stitched_image.save(output_filepath, format='TIFF', compression='tiff_deflate') # Use Pillow's deflate
                            print(f"Pillow 回退保存 TIFF 成功 (无地理信息)。")
                        else:
                            print(f"无法确定回退的 Pillow 模式，保存失败。")
                    except Exception as pillow_e:
                        print(f"错误: Pillow 回退保存 TIFF 也失败: {pillow_e}")

            # --- Use Pillow for other formats ---
            elif not is_output_tiff:
                 try:
                    print(f"正在使用 Pillow 保存 {output_format_string}: {output_filepath}")
                    if pillow_mode is None:
                        print(f"错误: 无法确定 Pillow 模式来保存非 TIFF 文件 '{output_filepath}'。")
                        continue # Skip saving this file

                    # Ensure canvas dtype is compatible with Pillow mode before creating image
                    # This is a basic check, more sophisticated checks might be needed
                    pillow_compatible = False
                    if pillow_mode == 'L' and canvas.dtype == np.uint8: pillow_compatible = True
                    elif pillow_mode == 'RGB' and canvas.dtype == np.uint8: pillow_compatible = True
                    elif pillow_mode == 'RGBA' and canvas.dtype == np.uint8: pillow_compatible = True
                    elif pillow_mode == 'I;16' and canvas.dtype == np.uint16: pillow_compatible = True # Pillow supports uint16 via I;16
                    elif pillow_mode == 'I' and canvas.dtype == np.int32: pillow_compatible = True
                    elif pillow_mode == 'F' and canvas.dtype == np.float32: pillow_compatible = True
                    # Add other modes/dtypes if necessary

                    if not pillow_compatible:
                         print(f"错误: NumPy 数组类型 ({canvas.dtype}) 与确定的 Pillow 模式 '{pillow_mode}' 不兼容。无法使用 Pillow 保存 '{output_filepath}'。")
                         continue


                    stitched_image = Image.fromarray(canvas, mode=pillow_mode)

                    save_options = {}
                    img_to_save = stitched_image # Potentially converted image

                    if output_format_string == 'JPEG':
                        save_options['quality'] = 95
                        # JPEG doesn't support alpha or other complex modes well
                        if img_to_save.mode not in ['L', 'RGB']:
                            print(f"   注意: 将图像从模式 {img_to_save.mode} 转换为 RGB 以保存为 JPEG。")
                            try:
                                img_to_save = img_to_save.convert('RGB')
                            except Exception as conv_e:
                                print(f"   错误: 转换为 RGB 失败: {conv_e}。无法保存为 JPEG。")
                                continue # Skip saving
                    elif output_format_string == 'PNG':
                        save_options['compress_level'] = 6 # 0 (no compression) to 9 (max)

                    img_to_save.save(output_filepath, format=output_format_string, **save_options)
                    print(f"Pillow 保存 {output_format_string} 成功!")
                 except Exception as e:
                    print(f"错误: 使用 Pillow 保存 '{output_filepath}' 时失败: {e}")

            # This case means it was TIFF but profile reading failed earlier
            elif is_output_tiff and first_tile_profile is None:
                 print(f"错误: 尝试保存 TIFF，但未能从第一个切片获取 Profile 信息，且 Pillow 模式也未能确定。无法保存 '{output_filepath}'。")

        else:
            print(f"没有成功处理任何切片，无法为标识符 '{identifier}' 生成拼接图像。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="拼接遵循 'name_xN_yM.ext' 命名规则的图像切片。"
                    "优先使用 Rasterio 处理和保存 TIFF 以尝试保留地理信息。"
                    "对于其他格式 (JPG, PNG)，使用 Pillow。",
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument("input_dir", help="包含图像切片的文件夹路径。")
    parser.add_argument("output_dir", help="保存拼接后图像的文件夹路径。")
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    stitch_images(input_path, output_path)

    print("\n脚本执行完毕。")