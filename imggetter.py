import os
import io
import csv
import datetime as dt
from typing import Optional, Dict, List

import typer
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image, ExifTags
import geopandas as gpd

app = typer.Typer(add_completion=False)

Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTS = (".tif", ".tiff", ".img", ".png", ".jpg", ".jpeg", ".shp")

# ---------------- EXIF GPS helper (for JPG/PNG with GPS) ---------------- #

def _exif_get_gps(img: Image.Image):
    try:
        exif = img._getexif()  # type: ignore[attr-defined]
    except Exception:
        exif = None
    if not exif:
        return None
    tagmap = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    gps = tagmap.get("GPSInfo")
    if not gps:
        return None

    def _r2d(values, ref):
        try:
            d = values[0][0] / values[0][1]
            m = values[1][0] / values[1][1]
            s = values[2][0] / values[2][1]
            deg = d + m / 60 + s / 3600
            if ref in ["S", "W"]:
                deg = -deg
            return deg
        except Exception:
            return None

    lat_vals, lat_ref = gps.get(2), gps.get(1)
    lon_vals, lon_ref = gps.get(4), gps.get(3)
    lat = _r2d(lat_vals, lat_ref) if lat_vals and lat_ref else None
    lon = _r2d(lon_vals, lon_ref) if lon_vals and lon_ref else None
    if lat is not None and lon is not None:
        return (lat, lon)
    return None

# ---------------- Core extractors returning a dict ---------------- #

def extract_info(file_path: str) -> Optional[Dict]:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in (".tif", ".tiff", ".img"):
        try:
            with rasterio.open(file_path) as ds:
                transform = ds.transform
                width, height = ds.width, ds.height
                x_resolution = transform.a
                y_resolution = abs(transform.e)
                crs = ds.crs
                crs_info = crs.to_string() if crs else "未定义 CRS"
                units = "未知单位"
                if crs:
                    try:
                        units = crs.linear_units or crs.name  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    if units == "unknown":
                        units = "未知单位"
                bounds = ds.bounds  # (left, bottom, right, top)
                # 转成 WGS84（若能）
                lon_min = lat_min = lon_max = lat_max = None
                if crs:
                    try:
                        l, b, r, t = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)
                        lon_min, lat_min, lon_max, lat_max = l, b, r, t
                    except Exception:
                        pass
                return {
                    "path": file_path,
                    "type": "raster",
                    "crs": crs_info,
                    "units": units,
                    "dtype": ",".join(ds.dtypes),
                    "bands": ds.count,
                    "width_px": width,
                    "height_px": height,
                    "x_resolution": x_resolution,
                    "y_resolution": y_resolution,
                    "x_min": bounds.left,
                    "y_min": bounds.bottom,
                    "x_max": bounds.right,
                    "y_max": bounds.top,
                    "lon_min": lon_min,
                    "lat_min": lat_min,
                    "lon_max": lon_max,
                    "lat_max": lat_max,
                }
        except Exception as e:
            typer.echo(f"处理影像文件时发生错误 '{file_path}': {e}")
            return None

    elif ext in (".png", ".jpg", ".jpeg"):
        try:
            with Image.open(file_path) as im:
                width, height = im.size
                mode = im.mode
                dpi = im.info.get("dpi")
                if isinstance(dpi, tuple) and len(dpi) == 2:
                    dpi_x, dpi_y = dpi
                else:
                    dpi_x = dpi_y = None
                channels = {
                    '1': 1, 'L': 1, 'RGB': 3, 'RGBA': 4, 'CMYK': 4, 'P': 1,
                }.get(mode, 'Unknown')
                gps = _exif_get_gps(im)
            rec = {
                "path": file_path,
                "type": "image",
                "mode": mode,
                "channels": channels,
                "width_px": width,
                "height_px": height,
                "dpi_x": dpi_x,
                "dpi_y": dpi_y,
            }
            if gps:
                rec.update({"exif_lat": gps[0], "exif_lon": gps[1]})
            return rec
        except Exception as e:
            typer.echo(f"处理图片文件时发生错误 '{file_path}': {e}")
            return None

    elif ext == ".shp":
        try:
            gdf = gpd.read_file(file_path)
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            crs = gdf.crs
            crs_info = crs.to_string() if crs else "未定义 CRS"
            units = "未知单位"
            if crs:
                try:
                    units = crs.axis_info[0].unit_name  # type: ignore[index]
                except Exception:
                    pass
            lon_min = lat_min = lon_max = lat_max = None
            try:
                if crs:
                    g4326 = gdf.to_crs(4326)
                    lb = g4326.total_bounds
                    lon_min, lat_min, lon_max, lat_max = lb[0], lb[1], lb[2], lb[3]
            except Exception:
                pass
            return {
                "path": file_path,
                "type": "shapefile",
                "crs": crs_info,
                "units": units,
                "features": len(gdf),
                "x_min": bounds[0],
                "y_min": bounds[1],
                "x_max": bounds[2],
                "y_max": bounds[3],
                "lon_min": lon_min,
                "lat_min": lat_min,
                "lon_max": lon_max,
                "lat_max": lat_max,
            }
        except Exception as e:
            typer.echo(f"处理 Shapefile 文件时发生错误 '{file_path}': {e}")
            return None

    else:
        return None

# ---------------- CLI: 单文件或目录批量并导出 CSV ---------------- #

@app.command()
def run(
    file_path: str = typer.Option(..., '-f', '--file-path', help='输入文件或文件夹路径（已挂载的本地/网络路径）'),
    output_csv: Optional[str] = typer.Option(None, '-o', '--output', help='当输入为文件夹时导出 CSV 到该路径；不指定则自动生成'),
    recursive: bool = typer.Option(True, '--recursive/--no-recursive', help='是否递归扫描子目录'),
):
    # 目录：批处理
    if os.path.isdir(file_path):
        records: List[Dict] = []
        typer.echo('开始扫描文件夹…')
        for root, _, files in os.walk(file_path):
            for name in files:
                if name.lower().endswith(SUPPORTED_EXTS):
                    p = os.path.join(root, name)
                    try:
                        info = extract_info(p)
                        if info:
                            records.append(info)
                            typer.echo(f"已解析: {p}")
                    except Exception as ex:
                        typer.echo(f"失败: {p} -> {ex}")
            if not recursive:
                break
        if not records:
            typer.echo('未在该目录中解析到受支持的文件。')
            raise typer.Exit(code=1)
        # 输出 CSV
        if not output_csv:
            ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = os.path.abspath(f'geo_summary_{ts}.csv')
        # 统一字段顺序：按出现的 key 集合
        all_keys = []
        seen = set()
        for r in records:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(records)
        typer.echo(f"✅ 已生成 CSV: {output_csv}")
        raise typer.Exit()

    # 单文件：打印信息
    if os.path.isfile(file_path):
        info = extract_info(file_path)
        if info is None:
            typer.echo(f"不支持的文件类型: {os.path.splitext(file_path)[1]}")
            raise typer.Exit(code=1)
        typer.echo(f"文件: {file_path}")
        for k, v in info.items():
            if k == 'path':
                continue
            typer.echo(f"{k}: {v}")
        raise typer.Exit()

    typer.echo('错误：路径既不是文件也不是文件夹，请检查是否已挂载/权限是否可访问。')
    raise typer.Exit(code=1)


if __name__ == '__main__':
    app()