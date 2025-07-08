import typer
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import traceback

Image.MAX_IMAGE_PIXELS = None
RASTERIO_COMPRESSION = None

app = typer.Typer(help="一个为机器学习优化的图像切片工具。所有输出均为无压缩TIFF。")

@app.command()
def crop_image(
        img_path: Path = typer.Option(
            ..., '-i', '--img-path',
            help="输入影像的路径 (支持 .tif, .tiff, .png, .jpg, .jpeg)。",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
        output_dir: Path = typer.Option(
            ..., '-o', '--output-dir',
            help="输出裁剪图像的文件夹路径。",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
        tile_size: int = typer.Option(
            768, '--tile-size',
            help="裁剪块的方形边长 (像素), 默认为 768。"
        ),
):
    """
    将大图像（TIFF, PNG, JPG）切割成指定大小的方形瓦片。
    所有输出瓦片都将保存为无压缩的TIFF格式，以优化机器学习流程。
    """
    try:
        file_suffix = img_path.suffix.lower()
        is_geospatial = file_suffix in ['.tif', '.tiff']
        is_standard_image = file_suffix in ['.png', '.jpg', '.jpeg']

        width, height = 0, 0
        src = None
        img = None

        if is_geospatial:
            print(f"正在使用 Rasterio 打开地理空间图像: {img_path}")
            src = rasterio.open(img_path)
            width, height = src.width, src.height
        elif is_standard_image:
            print(f"正在使用 Pillow 打开标准图像: {img_path}")
            img = Image.open(img_path)
            width, height = img.size
        else:
            print(f"错误：不支持的文件格式 '{file_suffix}'。请输入 .tif, .png, 或 .jpg 文件。")
            raise typer.Exit(code=1)

        print(f"图像尺寸: {width} x {height}")
        print(f"目标瓦片尺寸: {tile_size} x {tile_size}")

        if tile_size <= 0:
            print("错误：瓦片大小必须是正整数。")
            raise typer.Exit(code=1)
        if width < tile_size or height < tile_size:
            print(f"错误：图像尺寸 ({width}x{height}) 小于目标瓦片尺寸 ({tile_size}x{tile_size})。无法生成完整瓦片。")
            raise typer.Exit(code=1)

        num_full_tiles_x = width // tile_size
        num_full_tiles_y = height // tile_size
        total_tiles_to_process = num_full_tiles_x * num_full_tiles_y

        if total_tiles_to_process == 0:
            print("错误：根据计算，无法生成任何完整瓦片。请检查瓦片尺寸设置。")
            raise typer.Exit(code=1)

        print(f"将生成 {num_full_tiles_x} x {num_full_tiles_y} = {total_tiles_to_process} 个无压缩TIFF瓦片")
        ignored_width = width % tile_size
        ignored_height = height % tile_size
        if ignored_width > 0 or ignored_height > 0:
            print(f"注意：图像右侧 {ignored_width} 像素和底部 {ignored_height} 像素将被忽略。")

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"瓦片将保存在: {output_dir}")
        base_filename = img_path.stem

        with tqdm(total=total_tiles_to_process, desc="正在切割无压缩瓦片", unit="tile") as pbar:
            for j in range(num_full_tiles_y):
                for i in range(num_full_tiles_x):
                    try:
                        output_filename = f"{base_filename}_tile_x{i}_y{j}.tif"

                        if is_geospatial:
                            col_off, row_off = i * tile_size, j * tile_size
                            window = Window(col_off, row_off, tile_size, tile_size)
                            data = src.read(window=window)

                            profile = src.profile.copy()
                            transform = src.window_transform(window)
                            profile.update({
                                'height': tile_size,
                                'width': tile_size,
                                'transform': transform,
                                'compress': RASTERIO_COMPRESSION,
                                'driver': 'GTiff',
                                'tiled': True,
                                'blockxsize': min(256, tile_size),
                                'blockysize': min(256, tile_size),
                            })
                            with rasterio.open(output_dir / output_filename, 'w', **profile) as dest:
                                dest.write(data)

                        elif is_standard_image:
                            left, upper = i * tile_size, j * tile_size
                            right, lower = left + tile_size, upper + tile_size
                            box = (left, upper, right, lower)

                            cropped_img = img.crop(box)
                            cropped_img.save(output_dir / output_filename, 'TIFF', compression='none')

                    except Exception as tile_err:
                        print(f"\n处理瓦片 (x={i}, y={j}) 时出错: {tile_err}")

                    pbar.update(1)

        print("\n所有无压缩瓦片处理完成!")
        print(f"请注意: 由于没有压缩，输出文件会占用较多磁盘空间。")

    except Exception as e:
        print(f"\n处理过程中发生意外错误: {e}")
        traceback.print_exc()
        raise typer.Exit(code=1)
    finally:
        if src:
            src.close()
        if img:
            img.close()


if __name__ == "__main__":
    app()