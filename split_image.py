import os
import typer
from osgeo import gdal
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def split_images(
        img_path: str = typer.Option(..., '-i', '--img-path', help="输入的图片文件路径 (.tif, .png, .jpg 等)"),
        output_dir: str = typer.Option(..., '-o', '--output-dir', help="输出裁剪图像的文件夹路径"),
        height: int = typer.Option(..., '--height', help="裁剪块的高度"),
        width: int = typer.Option(..., '--width', help="裁剪块的宽度"),
        basename: str = typer.Option(None, '--basename', help="输出图像的基础名称")
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if basename is None:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
    else:
        base_name = basename

    if img_path.lower().endswith(('.tif', '.tiff')):
        dataset = gdal.Open(img_path)
        if dataset is None:
            typer.echo(f"无法打开输入文件: {img_path}")
            raise typer.Exit(code=1)

        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        bands = dataset.RasterCount

        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        driver = gdal.GetDriverByName('GTiff')

        x_blocks = img_width // width
        y_blocks = img_height // height

        for x in range(x_blocks):
            for y in range(y_blocks):
                x_offset = x * width
                y_offset = y * height

                data = dataset.ReadAsArray(x_offset, y_offset, width, height)

                # 检查数据类型并转换为 uint16
                if data.dtype == np.uint32:
                    data = (data / np.iinfo(np.uint32).max * np.iinfo(np.uint16).max).astype(np.uint16)

                cropped_name = f"{base_name}_x{x}_y{y}.tif"
                cropped_output_path = os.path.join(output_dir, cropped_name)

                # 使用 LZW 压缩和像素交织
                output_dataset = driver.Create(
                    cropped_output_path,
                    width, height, bands, gdal.GDT_UInt16,
                    options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL']  # 设置像素交织和LZW压缩
                )

                # 设置裁剪图像的地理坐标变换
                new_transform = list(transform)
                new_transform[0] = transform[0] + x_offset * transform[1]
                new_transform[3] = transform[3] + y_offset * transform[5]
                output_dataset.SetGeoTransform(new_transform)
                output_dataset.SetProjection(projection)

                # 写入各个波段数据
                for b in range(bands):
                    output_dataset.GetRasterBand(b + 1).WriteArray(data[b])

                    # 设置波段的颜色解释
                    color_interp = dataset.GetRasterBand(b + 1).GetColorInterpretation()
                    output_dataset.GetRasterBand(b + 1).SetColorInterpretation(color_interp)

                output_dataset = None  # 关闭输出文件
                typer.echo(f'裁剪图像已保存至: {cropped_output_path}')

        dataset = None  # 关闭输入文件

    elif img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # For image formats like PNG, JPG, etc.
        with Image.open(img_path) as img:
            img_width, img_height = img.size

            x_blocks = img_width // width
            y_blocks = img_height // height

            for x in range(x_blocks):
                for y in range(y_blocks):
                    left = x * width
                    upper = y * height
                    right = left + width
                    lower = upper + height

                    cropped_img = img.crop((left, upper, right, lower))
                    cropped_name = f"{base_name}_x{x}_y{y}.png"
                    cropped_output_path = os.path.join(output_dir, cropped_name)

                    cropped_img.save(cropped_output_path)
                    typer.echo(f'裁剪图像已保存至: {cropped_output_path}')

    else:
        typer.echo(f"不支持的文件格式: {img_path}")
        raise typer.Exit(code=1)
