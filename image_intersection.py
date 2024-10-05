import os
import rasterio
from rasterio.windows import from_bounds
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
import typer

def crop_file(
        file_path: str = typer.Option(..., '-f', '--file-path', help="输入文件的路径，可以是影像、图片或 Shapefile 文件"),
        left: float = typer.Option(..., help="最左侧 X 坐标"),
        right: float = typer.Option(..., help="最右侧 X 坐标"),
        top: float = typer.Option(..., help="最顶部 Y 坐标"),
        bottom: float = typer.Option(..., help="最底部 Y 坐标")
):
    _, file_extension = os.path.splitext(file_path)

    bounds = (left, bottom, right, top)

    if file_extension.lower() in [".tif", ".img", ".tiff"]:
        with rasterio.open(file_path) as src:
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            data = src.read(window=window)
            transform = src.window_transform(window)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": src.driver,
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform
            })

            output_file = file_path.replace(file_extension, f"_cropped{file_extension}")
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(data)

        typer.echo(f"裁剪后的影像已保存为: {output_file}")

    elif file_extension.lower() == ".shp":
        gdf = gpd.read_file(file_path)
        bbox = box(left, bottom, right, top)
        cropped_gdf = gdf[gdf.intersects(bbox)]
        output_file = file_path.replace(".shp", "_cropped.shp")
        cropped_gdf.to_file(output_file)

        typer.echo(f"裁剪后的 Shapefile 已保存为: {output_file}")

    elif file_extension.lower() in [".png", ".jpg", ".jpeg"]:
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(file_path)
        width, height = image.size

        pixel_left = int(left)
        pixel_right = int(right)
        pixel_top = int(top)
        pixel_bottom = int(bottom)

        cropped_image = image.crop((pixel_left, pixel_top, pixel_right, pixel_bottom))
        output_file = file_path.replace(file_extension, f"_cropped{file_extension}")
        cropped_image.save(output_file)

        typer.echo(f"裁剪后的图片已保存为: {output_file}")

    else:
        typer.echo(f"不支持的文件类型: {file_extension}")
