import typer
from osgeo import gdal
import os

def convert_img_to_tiff(
        img_path: str = typer.Option(..., '-i', '--img-path', help="输入的 .img 或 .ige 文件路径"),
        output_dir: str = typer.Option(..., '-o', '--output-dir', help="输出裁剪图像的文件夹路径"),
        height: int = typer.Option(None, '--height', help="裁剪块的高度"),
        width: int = typer.Option(None, '--width', help="裁剪块的宽度"),
        basename: str = typer.Option(None, '--basename', help="输出图像的基础名称")
):
    dataset = gdal.Open(img_path)
    if dataset is None:
        typer.echo(f"无法打开输入文件: {img_path}")
        raise typer.Exit(code=1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if basename is None:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
    else:
        base_name = basename

    if height is None or width is None:
        output_path = os.path.join(output_dir, f"{base_name}.tiff")
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.CreateCopy(output_path, dataset, 0)
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())
        output_dataset.SetProjection(dataset.GetProjection())
        typer.echo(f'图像已成功转换为 TIFF 格式，保存至: {output_path}')
    else:
        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        bands = dataset.RasterCount

        transform = dataset.GetGeoTransform()

        driver = gdal.GetDriverByName('GTiff')

        x_blocks = (img_width + width - 1) // width
        y_blocks = (img_height + height - 1) // height

        for x in range(x_blocks):
            for y in range(y_blocks):
                x_offset = x * width
                y_offset = y * height
                x_size = min(width, img_width - x_offset)
                y_size = min(height, img_height - y_offset)

                data = dataset.ReadAsArray(x_offset, y_offset, x_size, y_size)

                cropped_name = f"{base_name}_x{x}_y{y}.tif"
                cropped_output_path = os.path.join(output_dir, cropped_name)

                output_dataset = driver.Create(cropped_output_path, x_size, y_size, bands, gdal.GDT_Byte)

                new_transform = list(transform)
                new_transform[0] = transform[0] + x_offset * transform[1]
                new_transform[3] = transform[3] + y_offset * transform[5]
                output_dataset.SetGeoTransform(new_transform)
                output_dataset.SetProjection(dataset.GetProjection())
                for b in range(bands):
                    output_dataset.GetRasterBand(b + 1).WriteArray(data[b])
                output_dataset = None
                typer.echo(f'裁剪图像已保存至: {cropped_output_path}')

    dataset = None
