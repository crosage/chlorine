import typer
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

def apply_mask(
    img_path: str = typer.Option(..., '-i', '--img-path', help="输入影像的路径"),
    shapefile_path: str = typer.Option(..., '-s', '--shapefile-path', help="Shapefile 文件的路径"),
    output_path: str = typer.Option(..., '-o', '--output-path', help="输出图像的路径")
):
    with rasterio.open(img_path) as src:
        image = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    geometries = [geom for geom in gdf.geometry]

    mask_shape = (int(height), int(width))

    mask_raster = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=mask_shape,
        transform=transform,
        fill=0,
        dtype='uint16'
    )

    data_4ch = image[:4, :, :].astype(np.uint16)
    mask_4d = np.stack([mask_raster] * 4, axis=0)
    zeros = np.zeros_like(data_4ch, dtype=np.uint16)
    meta.update({"driver": "GTiff", "count": 4})
    data_masked = np.where(mask_4d == 0, zeros, data_4ch)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data_masked)

    typer.echo(f"图像处理完成，已保存为 '{output_path}'")