import typer
import geopandas as gpd
import numpy as np

def extract_polygon(
    shapefile_path: str = typer.Option(..., '-f', '--shapefile-path', help="Shapefile 文件的路径"),
    output_path: str = typer.Option("polygon_points.npy", '-o', '--output-path', help="输出的 NumPy 文件路径")
):
    gdf = gpd.read_file(shapefile_path)

    polygon_points_list = []

    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.coords.xy
            coords = np.vstack((x, y)).T
            polygon_points_list.append(coords)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom:
                x, y = poly.exterior.coords.xy
                coords = np.vstack((x, y)).T
                polygon_points_list.append(coords)
        else:
            typer.echo(f"不支持的几何类型: {geom.geom_type}")

    polygon_points_array = np.array(polygon_points_list, dtype=object)
    np.save(output_path, polygon_points_array)
    typer.echo(f"多边形坐标已保存至: {output_path}")
    for idx, coords in enumerate(polygon_points_array):
        typer.echo(f"第 {idx+1} 个闭合区域的坐标：")
        typer.echo(coords)