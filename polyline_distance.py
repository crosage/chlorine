import hashlib
import os
from math import dist
import numpy as np
import geopandas as gpd
import json
import yaml
from matplotlib import pyplot as plt
from shapely.geometry import Polygon,Point, LineString,MultiLineString,MultiPoint
from shapely.ops import nearest_points, linemerge, substring
import random
# line.interpolate用于找一条线上指定距离的点
class Polyline:
    def __init__(self, id, points):
        self.id = id
        self.points = points
        self.line = LineString(self.points)

    def __repr__(self):
        return f"Polyline(id={self.id}, points={self.points})"

    def length_between_points(self, point1, point2):
        point1 = Point(point1)
        point2 = Point(point2)

        projected_point1 = self.line.project(point1)
        projected_point2 = self.line.project(point2)

        if projected_point1 > projected_point2:
            projected_point1, projected_point2 = projected_point2, projected_point1

        subline = self.line.interpolate(projected_point1).coords[:] + self.line.interpolate(projected_point2).coords[:]
        subline = LineString(subline)

        return self.line.project(Point(subline.coords[0]), normalized=True) * self.line.length - self.line.project(
            Point(subline.coords[-1]), normalized=True) * self.line.length

    def distance_to_point(self, point):
        return point.distance(self.line)

    def estimate_length(self):
        total_length = 0.0
        for i in range(len(self.points) - 1):
            point1 = self.points[i]
            point2 = self.points[i + 1]
            distance = point1.distance(point2)
            total_length += distance

        return total_length


    def smooth_with_boundary(self, boundary, interval, new_id=None):
        """
        平滑曲线，每隔 interval 米取一个点，并确保线段不超出边界。
        如果线段与边界相交，逐步细化该线段。

        :param boundary: 边界线 (Polygon)。
        :param interval: 平滑间隔距离（米）。
        :param new_id: 新 Polyline 的 ID。
        :return: 符合边界约束的新的平滑 Polyline 对象。
        """
        if interval <= 0:
            raise ValueError("Interval must be greater than 0.")

        smooth_points = [self.line.interpolate(i * interval) for i in range(int(self.line.length // interval) + 1)]

        refined_points = [smooth_points[0]]

        for i in range(1, len(smooth_points)):
            segment = LineString([refined_points[-1], smooth_points[i]])

            # 检查当前段是否与边界相交
            while not boundary.contains(segment):
                mid_point = segment.interpolate(0.5, normalized=True)  # 取中点
                refined_points.append(mid_point)
                segment = LineString([refined_points[-1], smooth_points[i]])

            refined_points.append(smooth_points[i])

        new_polyline = Polyline(id=new_id if new_id else self.id, points=refined_points)
        return new_polyline


class ClosedShape:
    def __init__(self, intersections, work_line_1, work_line_2, tangent_line_1, tangent_line_2,polygon):
        self.intersections = intersections
        self.work_line_1 = work_line_1
        self.work_line_2 = work_line_2
        self.tangent_line_1 = tangent_line_1
        self.tangent_line_2 = tangent_line_2
        self.polygon = polygon

    def contains_point(self, point):
        return self.polygon.contains(point)

    def __repr__(self):
        return f"ClosedShape(intersections={self.intersections})"


def generate_infinite_normals_on_linestring_with_polyline(line, work_polyline, interval=100):
    line_length = line.length
    points_with_normals = []

    for distance in range(0, int(line_length) + 1, interval):
        print(f"目前distance:{distance} ")
        point = line.interpolate(distance)

        if distance == 0:
            next_point = line.interpolate(distance + 1)
            tangent_vector = np.array([next_point.x - point.x, next_point.y - point.y])
        elif distance >= line_length:
            prev_point = line.interpolate(distance - 1)
            tangent_vector = np.array([point.x - prev_point.x, point.y - prev_point.y])
        else:
            prev_point = line.interpolate(distance - 1)
            next_point = line.interpolate(distance + 1)
            tangent_vector = np.array([next_point.x - prev_point.x, next_point.y - prev_point.y])

        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        offset_start = np.array([point.x, point.y]) - (normal_vector * 1e6)
        offset_end = np.array([point.x, point.y]) + (normal_vector * 1e6)
        infinite_normal_line = LineString([offset_start, offset_end])

        intersection = work_polyline.intersection(infinite_normal_line)

        if intersection.is_empty or not hasattr(intersection, "geoms") or len(intersection.geoms) < 2:
            continue

        sorted_points = sorted(intersection.geoms, key=lambda p: p.distance(point))
        normal_line = LineString(sorted_points[:2])
        points_with_normals.append((point, normal_line))

    print(f"初步生成的法线数量: {len(points_with_normals)}")

    def remove_crossing_normals(points_with_normals):
        while True:
            crossings = {}
            for i, (_, line1) in enumerate(points_with_normals):
                crossings[i] = 0
                for j, (_, line2) in enumerate(points_with_normals):
                    if i != j and line1.intersects(line2):
                        crossings[i] += 1

            max_cross_index = max(crossings, key=crossings.get)
            max_cross_count = crossings[max_cross_index]
            print(f"当前交点最多的法线索引: {max_cross_index}, 交点数量: {max_cross_count}")
            if max_cross_count == 0:
                break
            points_with_normals.pop(max_cross_index)
        return points_with_normals
    points_with_normals = remove_crossing_normals(points_with_normals)
    print(f"去除交叉后剩余的法线数量: {len(points_with_normals)}")
    return points_with_normals


def load_polylines_from_shp(file_path, ignore):


    gdf = gpd.read_file(file_path)
    print("原始 CRS:", gdf.crs)
    gdf = gdf.to_crs("EPSG:32650")
    print("转换后的 CRS:", gdf.crs)

    polylines = []
    for idx, geom in enumerate(gdf.geometry):
        print(f"Geometry {idx} type: {type(geom)}")
        if idx == 8 and ignore:
            continue
        if isinstance(geom, LineString):
            points = [Point(coord) for coord in geom.coords]
            polyline = Polyline(id=idx, points=points)
            polylines.append(polyline)
        elif isinstance(geom, MultiLineString):
            for sub_idx, line in enumerate(geom.geoms):
                points = [Point(coord) for coord in line.coords]
                polyline = Polyline(id=f"{idx}_{sub_idx}", points=points)
                polylines.append(polyline)
    return polylines

def plot_polyline(polyline):
    x_coords = [point.x for point in polyline.points]
    y_coords = [point.y for point in polyline.points]
    plt.plot(x_coords, y_coords, label=f'Polyline {polyline.id}')
    plt.scatter(x_coords, y_coords, s=10)


def plot_polylines_with_labels(polylines, show=True):
    plt.figure(figsize=(12, 12))
    min_x, min_y, max_x, max_y = None, None, None, None

    for polyline in polylines:
        x, y = polyline.line.xy
        min_x = min(min(x), min_x) if min_x is not None else min(x)
        max_x = max(max(x), max_x) if max_x is not None else max(x)
        min_y = min(min(y), min_y) if min_y is not None else min(y)
        max_y = max(max(y), max_y) if max_y is not None else max(y)

    plt.xlim(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x))
    plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))

    for polyline in polylines:
        x, y = polyline.line.xy
        color = (random.random(), random.random(), random.random())
        plt.plot(x, y, color=color, label=f'Polyline {polyline.id}')
        start_x, start_y = x[0], y[0]
        plt.text(start_x, start_y, f"{polyline.id}", fontsize=10, color=color,
                 ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Polylines with Labels")

    if show:
        plt.show()


def find_closest_polyline(current_end, polylines):
    min_distance = float('inf')
    closest_polyline = None
    closest_points = []
    for polyline in polylines:
        start_dist = dist((current_end.x, current_end.y), (polyline.points[0].x, polyline.points[0].y))
        end_dist = dist((current_end.x, current_end.y), (polyline.points[-1].x, polyline.points[-1].y))

        if start_dist < min_distance:
            min_distance = start_dist
            closest_polyline = polyline
            closest_points = polyline.points  # 正向

        if end_dist < min_distance:
            min_distance = end_dist
            closest_polyline = polyline
            closest_points = list(reversed(polyline.points))  # 反向

    return closest_polyline, closest_points


def find_starting_polyline(polylines):
    min_point = None
    starting_polyline = None
    starting_points = []

    for polyline in polylines:
        for point in [polyline.points[0], polyline.points[-1]]:
            if min_point is None or (point.x < min_point.x or (point.x == min_point.x and point.y < min_point.y)):
                min_point = point
                starting_polyline = polyline
                starting_points = polyline.points if point == polyline.points[0] else list(reversed(polyline.points))

    return starting_polyline, starting_points


def plot_polylines_with_labels_and_merged(polylines, merged_points=None, step=None):
    plt.figure(figsize=(12, 12))

    # 动态缩放范围
    min_x, min_y, max_x, max_y = None, None, None, None
    for polyline in polylines:
        x, y = zip(*[(point.x, point.y) for point in polyline.points])
        min_x = min(min(x), min_x) if min_x is not None else min(x)
        max_x = max(max(x), max_x) if max_x is not None else max(x)
        min_y = min(min(y), min_y) if min_y is not None else min(y)
        max_y = max(max(y), max_y) if max_y is not None else max(y)

    plt.xlim(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x))
    plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))

    merged_set = set(merged_points) if merged_points else set()

    # 绘制每个折线并添加标签
    for polyline in polylines:
        x, y = zip(*[(point.x, point.y) for point in polyline.points])
        color = (random.random(), random.random(), random.random())

        if not set(polyline.points).issubset(merged_set):
            # 只为未合并的折线绘制图例和标签
            plt.plot(x, y, color=color, linewidth=1, label=f'Polyline {polyline.id}')
            start_x, start_y = x[0], y[0]
            end_x, end_y = x[-1], y[-1]
            plt.scatter([start_x], [start_y], color=color, s=30, edgecolor='black', zorder=3)

            plt.text(start_x, start_y, f"{polyline.id}", fontsize=10, color=color,
                     ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        else:
            # 已合并的折线，不显示图例和标签
            plt.plot(x, y, color=color, linewidth=1)

    if merged_points:
        x, y = zip(*[(point.x, point.y) for point in merged_points])
        plt.plot(x, y, color='red', linewidth=1.5, label='Merged')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f'Polylines with Labels - Step {step}' if step is not None else 'Merged Polylines')
    plt.legend()
    plt.show()


def merge_polylines(polylines, show=True):
    if not polylines:
        return None

    starting_polyline, merged_points = find_starting_polyline(polylines)
    remaining_polylines = [p for p in polylines if p != starting_polyline]

    step = 0
    if show:
        plot_polylines_with_labels_and_merged(polylines, merged_points, step)

    while remaining_polylines:
        current_end = merged_points[-1]
        closest_polyline, closest_points = find_closest_polyline(current_end, remaining_polylines)
        print(f"最近的id为{closest_polyline.id} 当前end为{current_end} 端点为{closest_points[0]}和{closest_points[-1]}")
        merged_points.extend(closest_points)
        remaining_polylines.remove(closest_polyline)

        step += 1
        if show:
            plot_polylines_with_labels_and_merged(polylines, merged_points, step)

    merged_polyline = Polyline(id="merged", points=merged_points)
    return merged_polyline


def plot_work(work_polylines, merged_line, show=True):
    for i, polyline in enumerate(work_polylines):
        if len(polyline.points) < 2:
            continue

        start_point = Point(polyline.points[0])
        end_point = Point(polyline.points[-1])

        proj_start = merged_line.line.project(start_point)
        proj_end = merged_line.line.project(end_point)
        proj_start_point = merged_line.line.interpolate(proj_start)
        proj_end_point = merged_line.line.interpolate(proj_end)

        print(f"第 {i + 1} 个 polyline 投影起点坐标: {proj_start_point}")
        print(f"第 {i + 1} 个 polyline 投影终点坐标: {proj_end_point}")

        if proj_start is None or proj_end is None or proj_start != proj_start or proj_end != proj_end:
            print("警告：检测到无效的投影点。跳过该 polyline。")
            continue

        length1 = merged_line.length_between_points(start_point, end_point)
        length2 = merged_line.length_between_points(proj_start_point, proj_end_point)
        print(f"第 {i + 1} 个 polyline 的投影点之间的距离为: {length1:.2f} {length2:.2f}")
        if(length1<0):
            length1=length1*-1
        if show:
            plt.figure(figsize=(10, 6))
            # print(f"projstart_point={proj_start_point.x}")
            plt.text(proj_start_point.x,proj_start_point.y,f"length={length1}")
            plt.plot(*merged_line.line.xy, label="Merged Line", color='black')
            plt.plot(*polyline.line.xy, label="Polyline", color='green')
            plt.scatter(*start_point.xy, color='red', label='start')
            plt.scatter(*end_point.xy, color='blue', label='end')
            plt.scatter(*proj_start_point.xy, color='orange', label='project_start')
            plt.scatter(*proj_end_point.xy, color='purple', label='project_end')
            plt.plot([start_point.x, proj_start_point.x], [start_point.y, proj_start_point.y], color='orange',
                     linestyle='--')
            plt.plot([end_point.x, proj_end_point.x], [end_point.y, proj_end_point.y], color='purple', linestyle='--')

            all_x = [start_point.x, end_point.x, proj_start_point.x, proj_end_point.x]
            all_y = [start_point.y, end_point.y, proj_start_point.y, proj_end_point.y]

            plt.xlim(min(all_x) - 1000, max(all_x) + 1000)
            plt.ylim(min(all_y) - 1000, max(all_y) + 1000)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f"number {i + 1}  Polyline project")
            plt.legend()
            plt.savefig(fname=f"{i+1}")
            # plt.show()


def extract_subcurve(line, point1, point2, show=False):
    """
    从 LineString 中提取从 point1 到 point2 的子曲线，并可选择性地显示调试图形。

    :param line: LineString 对象。
    :param point1: 起始点 (Point 对象)。
    :param point2: 结束点 (Point 对象)。
    :param show: 是否显示调试图形 (布尔值)。
    :return: 提取的子曲线 (LineString 对象)。
    """
    try:
        # 计算点在折线上的距离
        distance1 = line.project(point1)
        distance2 = line.project(point2)
        print(f"Point1 在折线上的距离: {distance1}")
        print(f"Point2 在折线上的距离: {distance2}")

        # 确保 distance1 <= distance2
        start_distance = min(distance1, distance2)
        end_distance = max(distance1, distance2)

        print("开始提取子曲线")
        # 使用 substring 提取子曲线
        subcurve = substring(line, start_distance, end_distance)
        print(f"提取的子曲线长度: {subcurve.length}")

        if show:
            # 创建图形
            fig, ax = plt.subplots(figsize=(8, 6))

            # 绘制原始折线
            x, y = line.xy
            ax.plot(x, y, label='Original Line', color='blue')

            # 绘制子曲线
            sub_x, sub_y = subcurve.xy
            ax.plot(sub_x, sub_y, label='Subcurve', color='green', linewidth=2)

            # 绘制点1和点2
            ax.plot(point1.x, point1.y, marker='o', markersize=8, color='red', label='Point 1')
            ax.plot(point2.x, point2.y, marker='o', markersize=8, color='orange', label='Point 2')

            # 标注点的位置
            ax.annotate(f'Point1 ({point1.x}, {point1.y})', xy=(point1.x, point1.y),
                        xytext=(point1.x + 0.1, point1.y + 0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05))

            ax.annotate(f'Point2 ({point2.x}, {point2.y})', xy=(point2.x, point2.y),
                        xytext=(point2.x + 0.1, point2.y + 0.1),
                        arrowprops=dict(facecolor='orange', shrink=0.05))

            # 设置图形属性
            ax.set_title('Extracted Subcurve Debug Visualization')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.show()

        return subcurve

    except Exception as e:
        print(f"提取子曲线时发生错误: {e}")
        return LineString()

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon, MultiPoint

def plot_closed_shapes_with_polylines(center_normals, work_polyline, original_line, save, log=False):
    """
    给定中心点、原始中心线和工作折线，求出每个封闭的区域。
    :param center_normals: 中心点及法线数据。
    :param work_polyline: 工作折线（边界线）。
    :param original_line: 原始中心线（用于计算真实切线方向）。
    :param show: 是否展示结果。
    :param log: 是否绘制调试过程。
    :return: 封闭形状列表。
    """
    closed_shapes = []

    # 如果需要绘图，创建图像和坐标轴

    for i in range(len(center_normals) - 1):
        print(f"当前正在遍历 {i}")
        # 当前点与下一个点
        current_point = center_normals[i][0]
        next_point = center_normals[i + 1][0]

        # 当前法线与下一个法线的点
        current_normal = center_normals[i][1].coords
        next_normal = center_normals[i + 1][1].coords

        # 提取法线点
        current_p1 = Point(current_normal[0][0], current_normal[0][1])
        current_p2 = Point(current_normal[1][0], current_normal[1][1])
        next_p1 = Point(next_normal[0][0], next_normal[0][1])
        next_p2 = Point(next_normal[1][0], next_normal[1][1])

        # 确定中心线方向向量（以当前点指向下一点）
        dvec = (next_point.x - current_point.x, next_point.y - current_point.y)

        # 计算参考法线方向（dvec旋转90度，选择 (-dy, dx)）
        nref = (-dvec[1], dvec[0])

        # 定义一个函数判断某点相对于中心点的法线方向
        def classify_point(center, point, nref):
            # 向量从中心点到目标点
            v = (point.x - center.x, point.y - center.y)
            # 计算点积
            return v[0] * nref[0] + v[1] * nref[1]

        # 使用点积与法线方向判断上下方点
        if classify_point(current_point, current_p1, nref) > classify_point(current_point, current_p2, nref):
            current_above = current_p1
            current_below = current_p2
        else:
            current_above = current_p2
            current_below = current_p1

        if classify_point(next_point, next_p1, nref) > classify_point(next_point, next_p2, nref):
            next_above = next_p1
            next_below = next_p2
        else:
            next_above = next_p2
            next_below = next_p1

        # 提取工作折线的子曲线
        upper_segment = extract_subcurve(work_polyline, current_above, next_above, show=log)
        lower_segment = extract_subcurve(work_polyline, next_below,current_below, show=log)

        polygon_coords = [
            (current_above.x, current_above.y),
            *list(upper_segment.coords),
            (next_above.x, next_above.y),
            (next_below.x, next_below.y),
            *list(lower_segment.coords),
            (current_below.x, current_below.y),
            (current_above.x, current_above.y)
        ]
        polygon = Polygon(polygon_coords)

        closed_shapes.append(ClosedShape(
            intersections=polygon_coords,
            work_line_1=LineString([current_above, current_below]),
            work_line_2=LineString([next_above, next_below]),
            tangent_line_1=upper_segment,
            tangent_line_2=lower_segment,
            polygon=polygon
        ))

        if save:
            fig, ax = plt.subplots(figsize=(12, 12))

            # 获取当前闭合形状的边界框
            min_x, min_y, max_x, max_y = polygon.bounds
            print(f"最左侧{min_x} 最右侧{max_x}")
            # 设置一定的边距
            margin = 10
            x_range = max_x - min_x
            y_range = max_y - min_y
            ax.set_xlim(min_x - x_range * margin, max_x + x_range * margin)
            ax.set_ylim(min_y - y_range * margin, max_y + y_range * margin)

            # 绘制工作折线和原始中心线
            ax.plot(*work_polyline.xy, color="blue", linewidth=1, label="Work Polyline")
            ax.plot(*original_line.xy, color="gray", linestyle="--", linewidth=1, label="Original Centerline")

            for j, shape in enumerate(closed_shapes):
                hash_input = str(j).encode('utf-8')
                hash_digest = hashlib.md5(hash_input).hexdigest()
                color = '#' + hash_digest[:6]

                px, py = shape.polygon.exterior.xy
                ax.fill(px, py, color=color, alpha=0.5, label=f"Closed Shape {j}" if j == 0 else "")
                ax.plot(px, py, color="red", linewidth=0.7)

            hash_input = str(i).encode('utf-8')
            hash_digest = hashlib.md5(hash_input).hexdigest()
            color = '#' + hash_digest[:6]
            px, py = polygon.exterior.xy
            ax.fill(px, py, color=color, alpha=0.5, label=f"Closed Shape {i}")
            ax.plot(px, py, color="red", linewidth=0.7)

            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Closed Shape {i} Visualization")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            ax.legend()

            image_filename = os.path.join(save, f'closed_shape_{i}.png')
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')
            plt.close()

            if log:
                print(f"已保存图像: {image_filename}")



    return closed_shapes


def save_split_points_to_file(split_points, file_path, file_format="yaml"):
    """
    保存分割点结果
    """
    if file_format not in ["yaml", "json"]:
        raise ValueError("file_format must be 'yaml' or 'json'")

    serialized_results = [
        {
            "point": {"x": point.x, "y": point.y},
            "normal_line": [
                {"x": coord[0], "y": coord[1]} for coord in normal_line.coords
            ]
        }
        for point, normal_line in split_points
    ]

    with open(file_path, "w") as f:
        if file_format == "yaml":
            yaml.dump(serialized_results, f, default_flow_style=False)
        elif file_format == "json":
            json.dump(serialized_results, f, indent=4)


def load_split_points_from_file(file_path, is_yaml=True):
    """
    加载分割点文件结果
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) if is_yaml else json.load(f)

    split_points = [
        (
            Point(item["point"]["x"], item["point"]["y"]),
            LineString([(coord["x"], coord["y"]) for coord in item["normal_line"]])
        )
        for item in data
    ]

    return split_points

def save_closed_shapes_to_file(closed_shapes, file_path, file_format="yaml"):
    """
    保存 Closed Shapes 结果，支持异常处理，确保运行不中断
    """
    if file_format not in ["yaml", "json"]:
        raise ValueError("file_format must be 'yaml' or 'json'")

    serialized_shapes = []

    for i, shape in enumerate(closed_shapes):
        print(f"正在保存{i}")
        try:
            serialized_shape = {
                "intersections": [
                    {"x": point[0], "y": point[1]} for point in shape.intersections
                ],
                "work_line_1": [
                    {"x": coord[0], "y": coord[1]} for coord in shape.work_line_1.coords
                ],
                "work_line_2": [
                    {"x": coord[0], "y": coord[1]} for coord in shape.work_line_2.coords
                ],
                "tangent_line_1": [
                    {"x": coord[0], "y": coord[1]} for coord in shape.tangent_line_1.coords
                ],
                "tangent_line_2": [
                    {"x": coord[0], "y": coord[1]} for coord in shape.tangent_line_2.coords
                ],
                "polygon": [
                    {"x": coord[0], "y": coord[1]} for coord in shape.polygon.exterior.coords
                ],
            }
            serialized_shapes.append(serialized_shape)
        except Exception as e:
            print(f"Error serializing ClosedShape at index {i}: {e}")
            continue  # 跳过当前 ClosedShape，继续处理下一个

    # 保存结果到文件
    try:
        with open(file_path, "w") as f:
            if file_format == "yaml":
                yaml.dump(serialized_shapes, f, default_flow_style=False)
            elif file_format == "json":
                json.dump(serialized_shapes, f, indent=4)
        print(f"Closed Shapes saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving Closed Shapes to file: {e}")

def load_closed_shapes_from_file(file_path, is_yaml=True):
    """
    加载 Closed Shapes 文件结果
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) if is_yaml else json.load(f)

    closed_shapes = []
    for item in data:
        intersections = [Point(p["x"], p["y"]) for p in item["intersections"]]
        work_line_1 = LineString([(coord["x"], coord["y"]) for coord in item["work_line_1"]])
        work_line_2 = LineString([(coord["x"], coord["y"]) for coord in item["work_line_2"]])
        tangent_line_1 = LineString([(coord["x"], coord["y"]) for coord in item["tangent_line_1"]])
        tangent_line_2 = LineString([(coord["x"], coord["y"]) for coord in item["tangent_line_2"]])
        polygon = Polygon([(coord["x"], coord["y"]) for coord in item["polygon"]])

        closed_shape = ClosedShape(
            intersections=intersections,
            work_line_1=work_line_1,
            work_line_2=work_line_2,
            tangent_line_1=tangent_line_1,
            tangent_line_2=tangent_line_2,
            polygon=polygon,
        )
        closed_shapes.append(closed_shape)

    return closed_shapes


def plot_split_points_with_lines(split_points_file, centerline_file, boundary_file):
    """
    绘制分割点，并展示中心线和边界线
    """
    print("进入plot")
    # 加载中心线
    center_polylines = load_polylines_from_shp(centerline_file, False)
    merged_centerline = merge_polylines(center_polylines, False)

    # 加载边界线（南北岸线）
    boundary_polylines = load_polylines_from_shp(boundary_file, False)

    # 加载分割点
    if not os.path.exists(split_points_file):
        print(f"分割点文件 {split_points_file} 不存在。请检查路径或重新生成。")
        return
    split_points = load_split_points_from_file(split_points_file, is_yaml=True)

    print("加载结束")
    # 绘制分割点和线条
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 绘制中心线
    x, y = merged_centerline.line.xy
    plt.plot(x, y, label="中心线", color="blue", linewidth=2)

    # 绘制边界线
    for i, polyline in enumerate(boundary_polylines):
        x, y = polyline.line.xy
        plt.plot(x, y, label=f"边界线 {i}", color="green", linestyle="--", linewidth=1)
    import time

    # 获取分割点总数
    total_points = len(split_points)
    print(f"Total split points to process: {total_points}")

    # 计数器初始化
    count = 0
    max_lines = 100  # 默认只加载前100000根，或处理完整个数组

    # 记录开始时间
    start_time = time.time()

    for point, normal_line in split_points:
        # 绘制分割点
        plt.scatter(point.x, point.y, color="red", s=10,
                    label="Split Point" if "Split Point" not in ax.get_legend_handles_labels()[1] else "")

        # 绘制法线
        nx, ny = normal_line.xy
        plt.plot(nx, ny, color="orange", linestyle=":", linewidth=0.5,
                 label="Normal Line" if "Normal Line" not in ax.get_legend_handles_labels()[1] else "")

        count += 1  # 更新计数器

        # 定期计算进度和 ETA
        if count % 100 == 0 or count == total_points:
            # 当前时间和已用时间
            elapsed_time = time.time() - start_time
            avg_time_per_point = elapsed_time / count
            remaining_points = total_points - count
            remaining_time = avg_time_per_point * remaining_points  # 剩余时间估算

            # 转换为小时:分钟:秒格式
            eta_seconds = int(remaining_time)
            eta_minutes, eta_seconds = divmod(eta_seconds, 60)
            eta_hours, eta_minutes = divmod(eta_minutes, 60)

            # 进度百分比
            progress = (count / total_points) * 100

            # 打印进度和 ETA
            print(f"Progress: {count}/{total_points} ({progress:.2f}%), "
                  f"ETA: {eta_hours:02}:{eta_minutes:02}:{eta_seconds:02}")

        if count >= max_lines:  # 超过max_lines后停止绘制
            print(f"Stopped processing after {count} points due to max_lines limit.")
            break
    # 设置图例、标题和布局
    plt.xlabel("X ")
    plt.ylabel("Y ")
    plt.title("plot")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main(use_smoothing=True):
    """
    主函数
    :param use_smoothing: 是否使用平滑的中心线，True 表示使用平滑，False 表示使用原始中心线
    """
    # 中心线
    centerline_file = "D:/机器学习数据/中心线和南北岸线/中心线平滑.shp"
    polylines = load_polylines_from_shp(centerline_file, False)
    merged_line = merge_polylines(polylines, False)

    # 边界线
    boundary_file = "D:/机器学习数据/中心线和南北岸线/南北线_修改后.shp"
    work_polylines = load_polylines_from_shp(boundary_file, False)
    boundary_polygon = work_polylines[0].line.convex_hull

    # 是否进行平滑处理
    if use_smoothing:
        print("Using smoothed centerline...")
        centerline = merged_line.smooth_with_boundary(boundary=boundary_polygon, interval=1000, new_id="smoothed_centerline")
    else:
        print("Using original centerline...")
        centerline = merged_line

    # 分割点结果文件
    split_points_file = "split_points.yaml"
    if os.path.exists(split_points_file):
        print(f"Found existing split points file: {split_points_file}. Loading split points...")
        result = load_split_points_from_file(split_points_file, is_yaml=True)
    else:
        print("No existing split points file found. Running calculations...")
        result = generate_infinite_normals_on_linestring_with_polyline(centerline.line, work_polylines[0].line, interval=1000)
        save_split_points_to_file(result, split_points_file, file_format="yaml")
        print(f"Split points saved to {split_points_file}")

    # Closed Shapes 结果文件
    closed_shapes_file = "closed_shapes.yaml"
    if os.path.exists(closed_shapes_file):
        print(f"Found existing closed shapes file: {closed_shapes_file}. Loading closed shapes...")
        closed_shapes = load_closed_shapes_from_file(closed_shapes_file, is_yaml=True)
    else:
        print("No existing closed shapes file found. Generating closed shapes...")
        closed_shapes = plot_closed_shapes_with_polylines(result, work_polylines[0].line,centerline.line, save="D:\\code\\shpdealer\\result2")
        save_closed_shapes_to_file(closed_shapes, closed_shapes_file, file_format="yaml")
    # 开始绘制所有元素
    fig, ax = plt.subplots(figsize=(12, 12))

    print("绘制边界线")
    # 绘制边界线
    for i, boundary in enumerate(work_polylines):
        bx, by = boundary.line.xy
        ax.plot(bx, by, color="green", label="Boundary Line" if i == 0 else "", linestyle="--")

    print("中心线")
    # 绘制原始中心线（仅当平滑启用时）
    if use_smoothing:
        ox, oy = merged_line.line.xy  # 原始中心线
        ax.plot(ox, oy, color="gray", linewidth=1, linestyle="--", label="Original Centerline")

    # 绘制平滑后的中心线或原始中心线
    cx, cy = centerline.line.xy
    if use_smoothing:
        ax.plot(cx, cy, color="blue", linewidth=2, label="Smoothed Centerline")
    else:
        ax.plot(cx, cy, color="blue", linewidth=2, label="Original Centerline")

    # 绘制分割点及法线
    for point, normal_line in result:
        ax.scatter(point.x, point.y, color="red", s=10, label="Split Points" if "Split Points" not in ax.get_legend_handles_labels()[1] else "")
        nx, ny = normal_line.xy
        ax.plot(nx, ny, color="orange", linestyle=":", linewidth=0.5, label="Normal Line" if "Normal Line" not in ax.get_legend_handles_labels()[1] else "")

    print("绘制闭合形状")
    # 绘制闭合形状
    for i, shape in enumerate(closed_shapes):
        print(f"绘制{i}")
        px, py = shape.polygon.exterior.xy
        color = (random.random(), random.random(), random.random())
        ax.plot(px, py, color=color, label=f"Closed Shape {i}" if i == 0 else "")
        ax.fill(px, py, color=color, alpha=0.2)

    # 设置图例、标题与样式
    ax.legend()
    ax.set_title("Centerline, Boundary, Split Points, and Closed Shapes")
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 控制是否使用平滑
    use_smoothing = True
    main(use_smoothing)
