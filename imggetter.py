import os
import rasterio
from PIL import Image
from pyproj import Transformer
import geopandas as gpd
import typer

def process_file(
        file_path: str = typer.Option(..., '-f', '--file-path', help="输入文件的路径，可以是影像、图片或 Shapefile 文件")
):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in [".tif", ".img", ".tiff"]:
        with rasterio.open(file_path) as dataset:
            transform = dataset.transform
            width = dataset.width
            height = dataset.height

            # 提取影像的四角坐标
            left_most = transform[2]
            right_most = transform[2] + (width * transform[0])
            top_most = transform[5]
            bottom_most = transform[5] + (height * transform[4])

            # 获取影像的 CRS 信息
            crs = dataset.crs
            crs_info = crs.to_string() if crs else "未定义 CRS"

        typer.echo(f"影像文件信息: {file_path}")
        typer.echo(f"坐标参考系 (CRS): {crs_info}")
        typer.echo(f"Left-most X coordinate: {left_most}")
        typer.echo(f"Right-most X coordinate: {right_most}")
        typer.echo(f"Top-most Y coordinate: {top_most}")
        typer.echo(f"Bottom-most Y coordinate: {bottom_most}")

    elif file_extension.lower() in [".png", ".jpg", ".jpeg"]:
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(file_path)
        width, height = image.size

        typer.echo(f"图片文件信息: {file_path}")
        typer.echo(f"Width: {width}, Height: {height}")


    elif file_extension.lower() == ".shp":
        gdf = gpd.read_file(file_path)
        bounds = gdf.total_bounds
        left_most = bounds[0]
        right_most = bounds[2]
        top_most = bounds[3]
        bottom_most = bounds[1]
        # 获取 CRS 信息
        crs = gdf.crs
        crs_info = crs.to_string() if crs else "未定义 CRS"

        # 获取坐标单位
        if crs:
            try:
                units = crs.axis_info[0].unit_name
            except AttributeError:
                units = "未知单位"
        else:
            units = "未定义 CRS"
        typer.echo(f"Shapefile 文件信息: {file_path}")
        typer.echo(f"坐标参考系 (CRS): {crs_info}")
        typer.echo(f"坐标单位: {units}")
        typer.echo(f"最左侧位置: {left_most} {units}")
        typer.echo(f"最右侧位置: {right_most} {units}")
        typer.echo(f"最顶部 Y 坐标: {top_most} {units}")
        typer.echo(f"最底部 Y 坐标: {bottom_most} {units}")

    else:
        typer.echo(f"不支持的文件类型: {file_extension}")

