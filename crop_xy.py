import os
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import fiona
from shapely.geometry import box
import typer

app = typer.Typer()

def crop_raster(img_path: str, output_tif: str, bounds: tuple):
    """
    裁剪栅格文件并保存到新的文件
    """
    with rasterio.open(img_path) as src:
        # 通过 bounds 获取窗口
        window = from_bounds(*bounds, src.transform)
        # 读取裁剪窗口内的栅格数据
        out_image = src.read(window=window)
        out_transform = src.window_transform(window)

        # 更新元数据
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # 写入裁剪后的栅格文件
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"栅格文件裁剪并保存至: {output_tif}")


def crop_shapefile(shp_path: str, output_shp: str, bounds: tuple):
    """
    裁剪Shapefile文件并保存到新的文件
    """
    bbox = box(*bounds)  # 创建用于裁剪的边界框

    # 打开原始Shapefile文件
    with fiona.open(shp_path, "r") as src:
        # 使用源文件的schema和crs创建新的Shapefile
        with fiona.open(
                output_shp, "w",
                driver=src.driver,
                crs=src.crs,
                schema=src.schema
        ) as dest:
            for feature in src:
                geom = feature["geometry"]
                if box(*bounds).intersects(geom):
                    # 仅保留与裁剪框相交的几何体
                    dest.write({
                        'geometry': geom,
                        'properties': feature["properties"]
                    })

    print(f"Shapefile文件裁剪并保存至: {output_shp}")


def crop_file(
        img_path: str = typer.Option(None, '-i', '--img-path', help="输入的 .tif 文件路径"),
        shp_path: str = typer.Option(None, '-s', '--shp-path', help="输入的 .shp 文件路径"),
        output_dir: str = typer.Option(..., '-o', '--output-dir', help="输出裁剪文件的文件夹路径"),
        xmin: float = typer.Option(..., '--xmin', help="左边界 X 坐标"),
        xmax: float = typer.Option(..., '--xmax', help="右边界 X 坐标"),
        ymin: float = typer.Option(..., '--ymin', help="下边界 Y 坐标"),
        ymax: float = typer.Option(..., '--ymax', help="上边界 Y 坐标"),
        basename: str = typer.Option(None, '--basename', help="输出文件的基础名称")
):
    """
    裁剪输入的栅格文件或Shapefile，并保存到输出文件夹
    """
    bounds = (xmin, ymin, xmax, ymax)

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 裁剪栅格文件
    if img_path:
        if not basename:
            basename = os.path.splitext(os.path.basename(img_path))[0]
        output_tif = os.path.join(output_dir, f"{basename}_cropped.tif")
        crop_raster(img_path, output_tif, bounds)

    # 裁剪Shapefile
    if shp_path:
        if not basename:
            basename = os.path.splitext(os.path.basename(shp_path))[0]
        output_shp = os.path.join(output_dir, f"{basename}_cropped.shp")
        crop_shapefile(shp_path, output_shp, bounds)


if __name__ == "__main__":
    app()
