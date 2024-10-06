import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
from fiona.transform import transform_geom
import os
import typer


# 定义命令行工具
def convert_crs(
        img_path: str = typer.Option(None, '-i', '--img-path', help="输入的 .tif、.img 或 .ige 文件路径"),
        shp_path: str = typer.Option(None, '-s', '--shp-path', help="输入的 .shp 文件路径"),
        output_dir: str = typer.Option(..., '-o', '--output-dir', help="输出裁剪图像或矢量文件的文件夹路径"),
        dst_crs: str = typer.Option(..., '--dst-crs', help="目标坐标参考系的EPSG代码, 例如: 'EPSG:32649'"),
        basename: str = typer.Option(None, '--basename', help="输出图像或矢量文件的基础名称")
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if img_path:
        if not basename:
            basename = os.path.splitext(os.path.basename(img_path))[0]

        output_tif = os.path.join(output_dir, f"{basename}_converted.tif")

        with rasterio.open(img_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_tif, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )

        print(f"栅格文件转换完成: {output_tif}")

    if shp_path:
        if not basename:
            basename = os.path.splitext(os.path.basename(shp_path))[0]

        output_shp = os.path.join(output_dir, f"{basename}_converted.shp")

        with fiona.open(shp_path, 'r') as src:
            src_crs = src.crs
            with fiona.open(
                output_shp,
                'w',
                driver=src.driver,
                crs=dst_crs,
                schema=src.schema
            ) as dst:
                for feature in src:
                    transformed_geom = transform_geom(src_crs, dst_crs, feature['geometry'])
                    feature['geometry'] = transformed_geom
                    dst.write(feature)

        print(f"Shapefile文件转换完成: {output_shp}")


