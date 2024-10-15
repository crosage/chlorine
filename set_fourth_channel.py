import typer
from osgeo import gdal
import os
from glob import glob

def set_fourth_channel_to_one(
        input_dir: str = typer.Option(..., '-i', '--input-dir', help="输入TIFF文件的文件夹路径")
):
    tiff_files = glob(os.path.join(input_dir, "*.tif"))

    if not tiff_files:
        typer.echo(f"没有找到TIFF文件在目录: {input_dir}")
        raise typer.Exit(code=1)

    for input_file in tiff_files:
        dataset = gdal.Open(input_file, gdal.GA_Update)
        if not dataset:
            typer.echo(f"无法打开输入文件: {input_file}")
            continue

        original_channels = dataset.RasterCount
        typer.echo(f"处理文件 {input_file}，原始通道数: {original_channels}")

        if original_channels < 4:
            typer.echo(f"文件 {input_file} 少于 4 个通道，跳过处理")
            continue

        fourth_band = dataset.GetRasterBand(4)
        fourth_band.Fill(1)
        typer.echo(f"文件 {input_file} 的第四个通道已全部设置为 1")


        dataset = None

        typer.echo(f"文件已成功修改: {input_file}")


if __name__ == "__main__":
    typer.run(set_fourth_channel_to_one)
