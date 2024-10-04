import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image
import typer

def crop_within_sample_area(
    img_path: str = typer.Option(..., '-i', '--img-path', help="输入影像的路径"),
    shapefile_path: str = typer.Option(..., '-s', '--shapefile-path', help="矢量数据的shapefile路径"),
    output_path: str = typer.Option(..., '-o', '--output-path', help="输出图像的路径")
):

    with rasterio.open(img_path) as dataset:
        transform = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height

        left_most = transform[2]
        right_most = transform[2] + (width * transform[0])
        top_most = transform[5]
        bottom_most = transform[5] + (height * transform[4])

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    minx, maxx, miny, maxy = left_most, right_most, bottom_most, top_most
    target_width, target_height = width, height
    transform = from_bounds(minx, miny, maxx, maxy, target_width, target_height)

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
    img.save(output_path)

    print(f'Output image saved to: {output_path}')
