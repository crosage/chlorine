import os
import typer
import rasterio
from rasterio.windows import Window
from math import ceil
from tqdm import tqdm
from pathlib import Path
RASTERIO_COMPRESSION = None
app = typer.Typer()

@app.command()
def crop_image(
    img_path: Path = typer.Option(
        ..., '-i', '--img-path',
        help="输入影像的路径 (TIFF/GeoTIFF).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ..., '-o', '--output-dir',
        help="输出裁剪图像的文件夹路径.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    tile_size: int = typer.Option(
        768, '--tile-size',
        help="裁剪块的边长 (像素), 默认为 768."
    ),
):

    try:
        print(f"正在使用 Rasterio 打开图像: {img_path}")
        with rasterio.open(img_path) as src:
            width = src.width
            height = src.height
            profile = src.profile
            print(f"图像尺寸: {width} x {height}")
            print(f"目标瓦片尺寸: {tile_size} x {tile_size}")
            if width < tile_size or height < tile_size:
                 print(f"错误：图像尺寸 ({width}x{height}) 小于目标瓦片尺寸 ({tile_size}x{tile_size})。无法生成完整瓦片。")
                 raise typer.Exit(code=1)
            if tile_size <= 0:
                 print("错误：瓦片大小必须是正整数。")
                 raise typer.Exit(code=1)
            num_full_tiles_x = width // tile_size
            num_full_tiles_y = height // tile_size
            total_tiles_to_process = num_full_tiles_x * num_full_tiles_y

            if total_tiles_to_process == 0:
                print("错误：根据计算，无法生成任何完整瓦片。")
                raise typer.Exit(code=1)

            print(f"将只生成 {num_full_tiles_x} x {num_full_tiles_y} = {total_tiles_to_process} 个完整瓦片")
            ignored_width = width % tile_size
            ignored_height = height % tile_size
            if ignored_width > 0 or ignored_height > 0:
                print(f"注意：图像右侧 {ignored_width} 像素和底部 {ignored_height} 像素将被忽略。")
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"瓦片将保存在: {output_dir}")
            base_filename = img_path.stem
            with tqdm(total=total_tiles_to_process, desc="正在切割完整瓦片", unit="tile") as pbar:
                for j in range(num_full_tiles_y):
                    for i in range(num_full_tiles_x):
                        col_off = i * tile_size
                        row_off = j * tile_size
                        window = Window(col_off, row_off, tile_size, tile_size)

                        try:
                            data = src.read(window=window)
                            out_profile = profile.copy()
                            out_transform = src.window_transform(window)

                            out_profile.update({
                                'height': tile_size,
                                'width': tile_size,
                                'transform': out_transform,
                                'compress': RASTERIO_COMPRESSION,
                                'driver': 'GTiff',
                                'tiled': True,
                                'blockxsize': 256,
                                'blockysize': 256
                            })

                            output_filename = f"{base_filename}_tile_x{i}_y{j}.tif"
                            output_path = output_dir / output_filename

                            with rasterio.open(output_path, 'w', **out_profile) as dest:
                                dest.write(data)

                        except Exception as tile_err:
                            print(f"\n处理瓦片 (x={i}, y={j}) 时出错: {tile_err}")

                        pbar.update(1)

        print("\n所有完整瓦片处理完成!")

    except rasterio.RasterioIOError as rio_err:
        print(f"错误: Rasterio无法读取文件 {img_path}. 文件可能损坏或格式不受支持.")
        print(f"Rasterio 错误信息: {rio_err}")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 {img_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"处理图像时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
