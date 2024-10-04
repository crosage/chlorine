import typer
import geopandas as gpd
from shapely.geometry import MultiPoint, box
import alphashape
import numpy as np

def create_concave_hull_polygon(
    shapefile_path: str = typer.Option(..., '-s', '--shapefile-path', help="输入的Shapefile文件路径"),
    output_shapefile_path: str = typer.Option(..., '-o', '--output-shapefile-path', help="输出的Shapefile文件路径"),
    left: float = typer.Option(None, '--left', help="最左边的X坐标"),
    right: float = typer.Option(None, '--right', help="最右边的X坐标"),
    top: float = typer.Option(None, '--top', help="最上边的Y坐标"),
    bottom: float = typer.Option(None, '--bottom', help="最下边的Y坐标"),
):

    gdf = gpd.read_file(shapefile_path)

    if None not in [left, right, top, bottom]:
        bounding_box = box(left, bottom, right, top)
        gdf = gdf.clip(bounding_box)

    points = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        elif geom.geom_type == 'Point':
            points.append((geom.x, geom.y))
        elif geom.geom_type == 'MultiPoint':
            points.extend([(pt.x, pt.y) for pt in geom.geoms])
        elif geom.geom_type in ['LineString', 'LinearRing']:
            points.extend(list(geom.coords))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                points.extend(list(line.coords))
        elif geom.geom_type == 'Polygon':
            points.extend(list(geom.exterior.coords))
            for interior in geom.interiors:
                points.extend(list(interior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                points.extend(list(polygon.exterior.coords))
                for interior in polygon.interiors:
                    points.extend(list(interior.coords))
        else:
            raise ValueError(f"不支持的几何类型: {geom.geom_type}")

    if len(points) < 3:
        raise ValueError("点的数量不足以生成凹壳。")
    print(f"点的数量为{len(points)}")
    alpha = alphashape.optimizealpha(points)
    print(f"alpha为{alpha}")
    concave_hull = alphashape.alphashape(points, alpha)

    result_gdf = gpd.GeoDataFrame(geometry=[concave_hull], crs=gdf.crs)
    result_gdf.to_file(output_shapefile_path)

if __name__ == "__main__":
    typer.run(create_concave_hull_polygon)
