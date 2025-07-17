import os
import rasterio
from PIL import Image
import geopandas as gpd
import typer


def process_file(
        file_path: str = typer.Option(..., '-f', '--file-path', help="输入文件的路径，可以是影像、图片或 Shapefile 文件")
):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in [".tif", ".img", ".tiff"]:
        try:
            with rasterio.open(file_path) as dataset:
                transform = dataset.transform
                width = dataset.width
                height = dataset.height

                # 获取空间分辨率
                # transform.a 是X方向的像素大小
                # abs(transform.e) 是Y方向的像素大小（取绝对值因为通常为负）
                x_resolution = transform.a
                y_resolution = abs(transform.e)  # Y方向通常是负值，取绝对值表示大小

                # 获取坐标单位
                crs_units = "未知单位"
                if dataset.crs:
                    try:
                        # 尝试获取线性单位（例如米、度）
                        crs_units = dataset.crs.linear_units
                        if crs_units is None:  # 如果线性单位为空，尝试获取名称
                            crs_units = dataset.crs.name
                    except AttributeError:
                        pass  # 如果没有linear_units属性，保持默认
                if crs_units == "unknown":
                    crs_units = "未知单位"

                num_bands = dataset.count
                left_most = transform[2]
                right_most = transform[2] + (width * transform[0])
                top_most = transform[5]
                bottom_most = transform[5] + (height * transform[4])
                data_types = dataset.dtypes
                crs = dataset.crs
                crs_info = crs.to_string() if crs else "未定义 CRS"

            typer.echo(f"影像文件信息: {file_path}")
            typer.echo(f"坐标参考系 (CRS): {crs_info}")
            typer.echo(f"坐标单位: {crs_units}")
            typer.echo(f"像素数据类型: {data_types}")
            typer.echo(f"波段数量: {num_bands}")
            typer.echo(f"图像宽度 (像素): {width}")
            typer.echo(f"图像高度 (像素): {height}")
            typer.echo(f"空间分辨率 (X方向): {x_resolution:.4f} {crs_units}/像素")
            typer.echo(f"空间分辨率 (Y方向): {y_resolution:.4f} {crs_units}/像素")
            if x_resolution == y_resolution:
                typer.echo(f"统一空间分辨率: {x_resolution:.4f} {crs_units}/像素")
            typer.echo(f"最左侧 X 坐标: {left_most}")
            typer.echo(f"最右侧 X 坐标: {right_most}")
            typer.echo(f"最顶部 Y 坐标: {top_most}")
            typer.echo(f"最底部 Y 坐标: {bottom_most}")
        except FileNotFoundError:
            typer.echo(f"错误：影像文件 '{file_path}' 不存在。")
        except rasterio.errors.RasterioIOError as e:
            typer.echo(f"错误：无法打开或读取影像文件 '{file_path}'。可能不是有效的栅格数据。错误信息: {e}")
        except Exception as e:
            typer.echo(f"处理影像文件时发生错误 '{file_path}': {e}")


    elif file_extension.lower() in [".png", ".jpg", ".jpeg"]:
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(file_path) as image:
                width, height = image.size
                mode = image.mode

                dpi = image.info.get('dpi')
                dpi_x, dpi_y = None, None
                if dpi and isinstance(dpi, tuple) and len(dpi) == 2:
                    dpi_x, dpi_y = dpi

                channels = {
                    '1': 1,
                    'L': 1,
                    'RGB': 3,
                    'RGBA': 4,
                    'CMYK': 4,
                    'P': 1,
                }.get(mode, 'Unknown')

                typer.echo(f"图片文件信息: {file_path}")
                typer.echo(f"像素宽度: {width}, 像素高度: {height}")
                typer.echo(f"模式 (Mode): {mode}, 通道数 (Channels): {channels}")
                if dpi_x is not None and dpi_y is not None:
                    typer.echo(f"DPI (水平): {dpi_x}, DPI (垂直): {dpi_y}")
                    if dpi_x > 0 and dpi_y > 0:
                        physical_width_inches = width / dpi_x
                        physical_height_inches = height / dpi_y
                        typer.echo(f"物理尺寸: {physical_width_inches:.2f} 英寸 x {physical_height_inches:.2f} 英寸")
                else:
                    typer.echo("该图片文件不包含 DPI 元数据。")
        except FileNotFoundError:
            typer.echo(f"错误：文件 '{file_path}' 不存在。")
        except Exception as e:
            typer.echo(f"处理图片文件时发生错误 '{file_path}': {e}")


    elif file_extension.lower() == ".shp":
        try:
            gdf = gpd.read_file(file_path)
            bounds = gdf.total_bounds
            left_most = bounds[0]
            right_most = bounds[2]
            top_most = bounds[3]
            bottom_most = bounds[1]
            crs = gdf.crs
            crs_info = crs.to_string() if crs else "未定义 CRS"

            units = "未知单位"
            if crs:
                try:
                    units = crs.axis_info[0].unit_name
                except (AttributeError, IndexError):
                    pass

            typer.echo(f"Shapefile 文件信息: {file_path}")
            typer.echo(f"坐标参考系 (CRS): {crs_info}")
            typer.echo(f"坐标单位: {units}")
            typer.echo(f"最左侧 X 坐标: {left_most} {units}")
            typer.echo(f"最右侧 X 坐标: {right_most} {units}")
            typer.echo(f"最顶部 Y 坐标: {top_most} {units}")
            typer.echo(f"最底部 Y 坐标: {bottom_most} {units}")
        except FileNotFoundError:
            typer.echo(f"错误：Shapefile 文件 '{file_path}' 不存在。")
        except Exception as e:
            typer.echo(f"处理 Shapefile 文件时发生错误 '{file_path}': {e}")

    else:
        typer.echo(f"不支持的文件类型: {file_extension}")


if __name__ == "__main__":
    typer.run(process_file)