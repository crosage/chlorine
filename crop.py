import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image
import typer
import os

def crop_within_sample_area(
        img_path: str = typer.Option(..., '-i', '--img-path', help="输入影像的路径"),
        shapefile_path: str = typer.Option(..., '-s', '--shapefile-path', help="矢量数据的shapefile路径"),
        output_dir: str = typer.Option(..., '-o', '--output-dir', help="输出裁剪图像的文件夹路径"),
        height: int = typer.Option(None, '--height', help="裁剪块的高度"),
        width: int = typer.Option(None, '--width', help="裁剪块的宽度"),
        basename: str = typer.Option(None, '--basename', help="输出图像的基础名称")
):
    with rasterio.open(img_path) as dataset:
        transform = dataset.transform
        crs = dataset.crs
        img_width = dataset.width
        img_height = dataset.height

        left_most = transform[2]
        right_most = transform[2] + (img_width * transform[0])
        top_most = transform[5]
        bottom_most = transform[5] + (img_height * transform[4])

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if height is None or width is None:
        target_width, target_height = img_width, img_height
        transform = from_bounds(left_most, bottom_most, right_most, top_most, target_width, target_height)

        raster = rasterize(
            ((geom, 1) for geom in gdf.geometry),
            out_shape=(target_height, target_width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        image[raster == 1] = [255, 255, 255]

        img = Image.fromarray(image)

        base_name = basename if basename else os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        img.save(output_path, format="PNG")
        print(f'图像已保存至: {output_path}')

    else:
        raster = rasterize(
            ((geom, 1) for geom in gdf.geometry),
            out_shape=(img_height, img_width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        x_blocks = img_width // width
        y_blocks = img_height // height

        for x in range(x_blocks):
            for y in range(y_blocks):
                x_offset = x * width
                y_offset = y * height

                if (x_offset + width <= img_width) and (y_offset + height <= img_height):
                    cropped_raster = raster[y_offset:y_offset + height, x_offset:x_offset + width]

                    image = np.zeros((height, width, 3), dtype=np.uint8)
                    image[cropped_raster == 1] = [255, 255, 255]

                    img = Image.fromarray(image)

                    base_name = basename if basename else os.path.splitext(os.path.basename(img_path))[0]
                    cropped_name = f"{base_name}_x{x}_y{y}.png"
                    output_path = os.path.join(output_dir, cropped_name)
                    img.save(output_path, format="PNG")

                    print(f'裁剪图像已保存至: {output_path}')