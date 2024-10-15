import typer
from osgeo import gdal
import os
from glob import glob


def convert_to_uint16(
        input_dir: str = typer.Option(..., '-i', '--input-dir', help="输入TIFF文件的文件夹路径"),
        num_channels: int = typer.Option(4, '-c', '--channels', help="目标通道数（如4通道、5通道等）")
):
    tiff_files = glob(os.path.join(input_dir, "*.tif"))

    if not tiff_files:
        typer.echo(f"没有找到TIFF文件在目录: {input_dir}")
        raise typer.Exit(code=1)

    for input_file in tiff_files:
        dataset = gdal.Open(input_file)
        if not dataset:
            typer.echo(f"无法打开输入文件: {input_file}")
            continue

        original_channels = dataset.RasterCount
        typer.echo(f"处理文件 {input_file}，原始通道数: {original_channels}")

        driver = gdal.GetDriverByName('MEM')
        output_dataset = driver.Create(
            '',
            dataset.RasterXSize,
            dataset.RasterYSize,
            num_channels,
            gdal.GDT_UInt16
        )

        for i in range(1, min(original_channels, num_channels) + 1):
            band = dataset.GetRasterBand(i)
            data = band.ReadAsArray()
            output_dataset.GetRasterBand(i).WriteArray(data)

        for i in range(original_channels + 1, num_channels + 1):
            output_dataset.GetRasterBand(i).Fill(1)
        dataset = None
        temp_output_file = input_file + ".tmp.tif"

        gdal.Translate(temp_output_file, output_dataset, options=gdal.TranslateOptions(
            outputType=gdal.GDT_UInt16,
            creationOptions=["BIGTIFF=YES"]
        ))

        os.remove(input_file)
        os.rename(temp_output_file, input_file)

        typer.echo(f"文件已成功修改并替换原文件: {input_file}")


if __name__ == "__main__":
    typer.run(convert_to_uint16)
