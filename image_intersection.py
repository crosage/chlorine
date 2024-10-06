import os
import typer
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

def image_intersection(
    file1_path: str = typer.Option(..., '-f1', '--file1-path', help="第一个文件的路径，可以是影像或 Shapefile 文件"),
    file2_path: str = typer.Option(..., '-f2', '--file2-path', help="第二个文件的路径，可以是影像或 Shapefile 文件"),
    output_dir: str = typer.Option(..., '-o', '--output-dir', help="保存交集部分的文件夹路径")
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取文件扩展名并转换为小写
    ext1 = os.path.splitext(file1_path)[1].lower()
    ext2 = os.path.splitext(file2_path)[1].lower()

    # 判断文件类型
    vector_exts = ['.shp', '.geojson', '.gpkg']
    raster_exts = ['.tif', '.tiff', '.img', '.jpg', '.jpeg', '.png']

    def is_vector(ext):
        return ext in vector_exts

    def is_raster(ext):
        return ext in raster_exts

    # 读取第一个文件
    if is_vector(ext1):
        gdf1 = gpd.read_file(file1_path)
        crs1 = gdf1.crs
    elif is_raster(ext1):
        src1 = rasterio.open(file1_path)
        crs1 = src1.crs
    else:
        typer.echo(f"不支持的文件类型: {file1_path}")
        raise typer.Exit()

    # 读取第二个文件
    if is_vector(ext2):
        gdf2 = gpd.read_file(file2_path)
        crs2 = gdf2.crs
    elif is_raster(ext2):
        src2 = rasterio.open(file2_path)
        crs2 = src2.crs
    else:
        typer.echo(f"不支持的文件类型: {file2_path}")
        raise typer.Exit()

    # 统一坐标系到第一个文件的坐标系
    if is_vector(ext1) and is_vector(ext2):
        if crs1 != crs2:
            gdf2 = gdf2.to_crs(crs1)
        # 计算交集
        intersection = gpd.overlay(gdf1, gdf2, how='intersection')
        if intersection.empty:
            typer.echo("两个矢量文件没有交集")
            raise typer.Exit()
        # 保存交集部分
        output_path1 = os.path.join(output_dir, f"intersection{ext1}")
        intersection.to_file(output_path1)
        typer.echo(f"交集已保存到 {output_path1}")

    elif is_raster(ext1) and is_raster(ext2):
        # 如果投影不同，需重投影
        if crs1 != crs2:
            typer.echo("两个栅格文件的坐标系不同，正在重投影第二个文件...")
            dst_crs = crs1
            transform, width, height = calculate_default_transform(
                src2.crs, dst_crs, src2.width, src2.height, *src2.bounds)
            kwargs = src2.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            temp_reprojected = os.path.join(output_dir, 'temp_reprojected.tif')
            with rasterio.open(temp_reprojected, 'w', **kwargs) as dst:
                for i in range(1, src2.count + 1):
                    reproject(
                        source=rasterio.band(src2, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src2.transform,
                        src_crs=src2.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
            src2.close()
            src2 = rasterio.open(temp_reprojected)
            crs2 = src2.crs

        # 计算交集范围
        bounds1 = src1.bounds
        bounds2 = src2.bounds
        intersection_bounds = (
            max(bounds1.left, bounds2.left),
            max(bounds1.bottom, bounds2.bottom),
            min(bounds1.right, bounds2.right),
            min(bounds1.top, bounds2.top)
        )
        if intersection_bounds[0] >= intersection_bounds[2] or intersection_bounds[1] >= intersection_bounds[3]:
            typer.echo("两个栅格文件没有交集")
            raise typer.Exit()

        # 创建交集的几何
        intersection_geom = box(*intersection_bounds)

        # 裁剪并保存第一个栅格文件
        out_image1, out_transform1 = mask(src1, [intersection_geom], crop=True)
        out_meta1 = src1.meta.copy()
        out_meta1.update({
            "driver": src1.driver,
            "height": out_image1.shape[1],
            "width": out_image1.shape[2],
            "transform": out_transform1
        })
        output_path1 = os.path.join(output_dir, f"intersection{ext1}")
        with rasterio.open(output_path1, "w", **out_meta1) as dest:
            dest.write(out_image1)
        typer.echo(f"第一个栅格交集已保存到 {output_path1}")

        # 裁剪并保存第二个栅格文件
        out_image2, out_transform2 = mask(src2, [intersection_geom], crop=True)
        out_meta2 = src2.meta.copy()
        out_meta2.update({
            "driver": src2.driver,
            "height": out_image2.shape[1],
            "width": out_image2.shape[2],
            "transform": out_transform2
        })
        output_path2 = os.path.join(output_dir, f"intersection{ext2}")
        with rasterio.open(output_path2, "w", **out_meta2) as dest:
            dest.write(out_image2)
        typer.echo(f"第二个栅格交集已保存到 {output_path2}")

        src1.close()
        src2.close()

        # 删除临时文件
        temp_reprojected = os.path.join(output_dir, 'temp_reprojected.tif')
        if os.path.exists(temp_reprojected):
            os.remove(temp_reprojected)

    elif (is_vector(ext1) and is_raster(ext2)) or (is_raster(ext1) and is_vector(ext2)):
        # 确保矢量文件与栅格文件在同一坐标系
        if is_vector(ext1) and is_raster(ext2):
            vector_gdf = gdf1
            raster_src = src2
            vector_ext = ext1
            raster_ext = ext2
        else:
            vector_gdf = gdf2
            raster_src = src1
            vector_ext = ext2
            raster_ext = ext1

        if vector_gdf.crs != raster_src.crs:
            typer.echo("矢量文件和栅格文件的坐标系不同，正在重投影矢量文件...")
            vector_gdf = vector_gdf.to_crs(raster_src.crs)

        # 裁剪矢量文件到栅格文件范围
        raster_bounds = raster_src.bounds
        raster_geom = box(*raster_bounds)
        vector_intersection = vector_gdf[vector_gdf.intersects(raster_geom)]
        if vector_intersection.empty:
            typer.echo("矢量文件和栅格文件没有交集")
            raise typer.Exit()

        # 保存矢量交集部分
        output_path_vector = os.path.join(output_dir, f"intersection{vector_ext}")
        vector_intersection.to_file(output_path_vector)
        typer.echo(f"矢量交集已保存到 {output_path_vector}")

        # 裁剪栅格文件到矢量文件范围
        shapes = [feature["geometry"] for feature in vector_intersection.__geo_interface__["features"]]
        out_image, out_transform = mask(raster_src, shapes, crop=True)
        out_meta = raster_src.meta.copy()
        out_meta.update({
            "driver": raster_src.driver,
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        output_path_raster = os.path.join(output_dir, f"intersection{raster_ext}")
        with rasterio.open(output_path_raster, "w", **out_meta) as dest:
            dest.write(out_image)
        typer.echo(f"栅格交集已保存到 {output_path_raster}")

        raster_src.close()
    else:
        typer.echo("文件类型不支持")
        raise typer.Exit()
