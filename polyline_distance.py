"""
大体思路：
1. 首先读取边界线和中心线
2. 因为边界线之间存在重叠以及中间断点情况，需要首先合并成一整根边界线
3. 然后给定两个点让边界线从这两个点断开成为南岸北岸，并进行保存
4. 然后中心线进行平滑
5. 中心线每隔一段点选取点然后从南岸北岸两岸伸出线取出的两个点分别保存为南岸点和北岸点

"""


import hashlib
import os
from math import dist
import numpy as np
import geopandas as gpd
import json
import yaml
from shapely.geometry import Polygon,Point, LineString,MultiLineString,MultiPoint
from shapely.ops import nearest_points, linemerge, substring
import random
import matplotlib.pyplot as plt
from shapely.geometry import LineString

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


def generate_infinite_normals_on_linestring_with_polyline(line, north, south, interval=100):
    """
    生成多段线上的法线，使用 north 和 south 作为参考线来计算法线方向。

    :param line: 多段线 (LineString)。
    :param north: 北岸参考线 (LineString)，用于确定法线的方向。
    :param south: 南岸参考线 (LineString)，用于确定法线的方向。
    :param interval: 间隔距离（米），用于在多段线的每个点之间生成法线。
    :return: 返回包含法线的点与法线的元组列表。
    """
    line_length = line.length
    points_with_normals = []  # 用于存储每个点和对应的法线

    for distance in range(0, int(line_length) + 1, interval):
        try:
            point = line.interpolate(distance)

            # 计算切线方向
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

            # 计算法线方向，垂直于切线
            normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 单位化法线

            # 使用 north 和 south 线确定法线的最终方向
            if north.contains(point):  # 如果点在 north 线的北边
                normal_vector = -normal_vector  # 反转法线方向

            # 生成无限法线
            offset_start = np.array([point.x, point.y]) - (normal_vector * 1e6)
            offset_end = np.array([point.x, point.y]) + (normal_vector * 1e6)
            infinite_normal_line = LineString([offset_start, offset_end])

            # 计算法线与 north 和 south 的交点
            intersection_with_north = infinite_normal_line.intersection(north)
            intersection_with_south = infinite_normal_line.intersection(south)

            if intersection_with_north.is_empty and intersection_with_south.is_empty:
                continue

            print(f"当前距离: {distance} ")
            north_points = []
            if not intersection_with_north.is_empty:
                if intersection_with_north.geom_type == 'MultiPoint':
                    north_points.extend(intersection_with_north.geoms)
                else:
                    north_points.append(intersection_with_north)

            south_points = []
            if not intersection_with_south.is_empty:
                if intersection_with_south.geom_type == 'MultiPoint':
                    south_points.extend(intersection_with_south.geoms)
                else:
                    south_points.append(intersection_with_south)
            print(north_points, south_points)

            north_point = min(north_points, key=lambda p: p.distance(point)) if north_points else None
            south_point = min(south_points, key=lambda p: p.distance(point)) if south_points else None

            if north_point and south_point:
                normal_line = LineString([north_point, south_point])
                points_with_normals.append((point, normal_line))
            else:
                continue

        except Exception as e:
            print(f"在距离 {distance} 处发生异常: {e}")
            continue

    print(f"初步生成的法线数量: {len(points_with_normals)}")

    def remove_crossing_normals(points_with_normals):
        """
        去除相交的法线，只保留不交叉的法线。
        """
        while True:
            try:
                crossings = {}  # 用于记录每个法线交叉的次数
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
            except Exception as e:
                print(f"在去除交叉法线时发生异常: {e}")
                break

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
        distance1 = line.line_locate_point(point1)
        distance2 = line.line_locate_point(point2)
        print(f"Point1 在折线上的距离: {distance1}")
        print(f"Point2 在折线上的距离: {distance2}")

        print("开始提取子曲线")
        # 使用 substring 提取子曲线
        subcurve = substring(line, distance1, distance2)
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



def split_polyline_by_points(work_polyline, point1, point2, point1_index, point2_index, log=None):
    """
    根据给定的两个点，将工作折线切割为两部分，并根据 `log` 参数判断是否展示调试信息。
    :param work_polyline: 要切割的工作折线（LineString）
    :param point1: 第一个切割点（Point）
    :param point2: 第二个切割点（Point）
    :param log: 包含数字的列表，指定要绘制的调试信息项
    :return: 返回切割后的两部分（north_line 和 south_line）
    """
    # 找到切割点的坐标索引
    coords = list(work_polyline.coords)
    start_index = point1_index
    end_index = point2_index
    if start_index < end_index:
        north_coords = coords[start_index:end_index + 1]
        south_coords = coords[end_index:] + coords[:start_index + 1]
    else:
        north_coords = coords[start_index:] + coords[:end_index + 1]
        # 这里我需要着重说一下为什么这么写，看起来，直接start_index:   + :end_index+1 那肯定的end_index+1:start_index即可解决问题对吧，实际上这个数据存在几条线之间互相重合的部分，所以需要用下面这种方法去掉上面已求出的真值部分才可以求出正确的部分，数据实在是惊为天人
        south_coords = [coord for coord in coords if coord not in north_coords]
    north_line = LineString(north_coords)
    south_line = LineString(south_coords)

    if log:
        fig, ax = plt.subplots(figsize=(8, 8))
        if 1 in log:
            # 绘制原始工作折线
            x, y = work_polyline.xy
            ax.plot(x, y, label="Work Polyline", color="blue", linewidth=2)

        if 2 in log:
            # 绘制北岸部分
            x, y = north_line.xy
            ax.plot(x, y, label="North Line", color="green", linewidth=2)

        if 3 in log:
            # 绘制南岸部分
            x, y = south_line.xy
            ax.plot(x, y, label="South Line", color="orange", linewidth=2)

        if 4 in log:
            # 从 start_index 到末尾 (start_index:)
            if len(work_polyline.coords[start_index:]) > 0:
                start_x, start_y = zip(*work_polyline.coords[start_index:])
                ax.plot(start_x, start_y, label=f"From Start ({start_index}:)", color="purple", linestyle=":",
                        linewidth=2)

        if 5 in log:
            # 从 end_index 到末尾 (end_index:)
            if len(work_polyline.coords[end_index:]) > 0:
                end_x, end_y = zip(*work_polyline.coords[end_index:])
                ax.plot(end_x, end_y, label=f"From End ({end_index}:)", color="cyan", linestyle=":", linewidth=2)

        if 6 in log:
            # 从 end_index 到 start_index (end_index:start_index)
            if len(work_polyline.coords[end_index:] + work_polyline.coords[:start_index]) > 0:
                wrap_x, wrap_y = zip(*work_polyline.coords[end_index:] + work_polyline.coords[:start_index])
                ax.plot(wrap_x, wrap_y, label=f"From End to Start ({end_index}:{start_index})", color="brown",
                        linestyle=":", linewidth=2)

        if 7 in log:
            # 从 start_index 到 end_index (start_index:end_index)
            if len(work_polyline.coords[start_index:end_index]) > 0:
                part_x, part_y = zip(*work_polyline.coords[start_index:end_index])
                ax.plot(part_x, part_y, label=f"From Start to End ({start_index}:{end_index})", color="orange",
                        linestyle=":", linewidth=2)

        if 8 in log:
            # 从开头到 end_index (:end_index)
            if len(work_polyline.coords[:end_index]) > 0:
                head_x, head_y = zip(*work_polyline.coords[:end_index])
                ax.plot(head_x, head_y, label=f"Up to End (:{end_index})", color="magenta", linestyle=":", linewidth=2)

        if 9 in log:
            # 从开头到 start_index (:start_index)
            if len(work_polyline.coords[:start_index]) > 0:
                head_start_x, head_start_y = zip(*work_polyline.coords[:start_index])
                ax.plot(head_start_x, head_start_y, label=f"Up to Start (:{start_index})", color="pink", linestyle=":",
                        linewidth=2)

        ax.scatter([point1.x, point2.x], [point1.y, point2.y], color="red", zorder=5, label="Cutting Points")

        ax.set_title("Work Polyline and Split Lines")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()

        ax.set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.show()

    return north_line, south_line


def save_north_south_lines_to_json(north_line, south_line, filename):
    """
    将北线和南线保存为 JSON 文件
    :param north_line: 北线 LineString
    :param south_line: 南线 LineString
    :param filename: 保存的 JSON 文件路径
    """
    north_coords = list(north_line.coords)
    south_coords = list(south_line.coords)

    data = {
        'north_line': north_coords,
        'south_line': south_coords
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Lines saved to {filename}")

def load_north_south_lines_from_json(filename):
    """
    从 JSON 文件加载并恢复北线和南线为 LineString 对象
    :param filename: JSON 文件路径
    :return: 恢复的北线和南线 LineString 对象
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    north_coords = data['north_line']
    south_coords = data['south_line']

    north_line = LineString(north_coords)
    south_line = LineString(south_coords)

    return north_line, south_line


def plot_north_south_lines(north_line, south_line):
    """
    可视化北线和南线
    :param north_line: 北线 LineString
    :param south_line: 南线 LineString
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制北线
    x, y = north_line.xy
    ax.plot(x, y, label='North Line', linewidth=2)

    # 绘制南线
    x, y = south_line.xy
    ax.plot(x, y, label='South Line', linewidth=2)

    # 标题和图例
    ax.set_title('Visualization of North and South Lines')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

    # 网格和比例
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    plt.show()



def plot_closed_shapes_with_polylines(center_normals, north_line, south_line, save, log=False):
    """
    给定中心点、北岸和南岸折线，求出每个封闭的区域。
    :param center_normals: 中心点及法线数据。
    :param north_line: 北岸折线（上方折线）。
    :param south_line: 南岸折线（下方折线）。
    :param original_line: 原始中心线（用于计算真实切线方向）。
    :param save: 是否保存结果图形。
    :param log: 是否绘制调试过程。
    :return: 封闭形状列表。
    """
    closed_shapes = []
    for i in range(len(center_normals) - 1):
        print(f"当前正在遍历 {i}")

        # 当前点与下一个点
        current_point = center_normals[i][0]
        next_point = center_normals[i + 1][0]

        # 当前法线与下一个法线的点
        current_normal = center_normals[i][1].coords
        next_normal = center_normals[i + 1][1].coords

        # 提取法线点
        current_above = Point(current_normal[0][0], current_normal[0][1]) # 北
        current_below = Point(current_normal[1][0], current_normal[1][1]) # 南
        next_above = Point(next_normal[0][0], next_normal[0][1]) # 北
        next_below = Point(next_normal[1][0], next_normal[1][1]) # 南

        # 提取北岸（上方）和南岸（下方）的子曲线
        upper_segment = extract_subcurve(north_line, current_above, next_above, show=log)
        lower_segment = extract_subcurve(south_line, next_below, current_below, show=log)

        # 构建封闭多边形
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
            ax.plot(*north_line.xy, color="blue", linewidth=1, label="Work Polyline")
            ax.plot(*south_line.xy, color="red", linewidth=1, label="Work Polyline")

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
# 生成法线并绘制的函数
def plot_normals_and_lines(line, north, south, interval=100):
    # 调用生成法线的函数
    points_with_normals = generate_infinite_normals_on_linestring_with_polyline(line, north, south, interval)

    # 设置绘图
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 绘制多段线
    x, y = line.xy
    ax.plot(x, y, label="Polyline", color='blue', lw=2)

    # 绘制北岸和南岸参考线
    x_north, y_north = north.xy
    ax.plot(x_north, y_north, label="North Line", color='green', lw=2, linestyle='--')

    x_south, y_south = south.xy
    ax.plot(x_south, y_south, label="South Line", color='red', lw=2, linestyle='--')

    # 绘制生成的法线
    for point, normal_line in points_with_normals:
        x_norm, y_norm = normal_line.xy
        ax.plot(x_norm, y_norm, color='orange', lw=1)

        # 标记法线的交点
        ax.plot(point.x, point.y, 'ko', label="Intersection Point")

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Generated Normals on Polyline")
    plt.show()


from shapely.geometry import Point


def find_point_in_closed_shapes(point, closed_shapes):
    """
    判断一个点属于哪个 ClosedShape 的 Polygon 中。
    :param point: 要检查的点 (shapely.geometry.Point)
    :param closed_shapes: ClosedShape 对象的列表
    :return: 属于的 ClosedShape 的索引和对象，如果不存在则返回 None
    """
    for i, shape in enumerate(closed_shapes):
        if shape.polygon.contains(point):
            return i, shape  # 返回索引和对应的 ClosedShape 对象

    return None, None  # 如果没有找到，返回 None


def plot_all_ditches(polylines, log=True):
    """
    使用 matplotlib 展示所有的清沟（ditch）线条。

    :param polylines: 由 load_polylines_from_shp 返回的 Polyline 对象列表。
    :param log: 是否显示每条线的信息。
    """
    if not polylines:
        print("未提供有效的清沟线条数据！")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    for idx, polyline in enumerate(polylines):
        x = [point.x for point in polyline.points]
        y = [point.y for point in polyline.points]

        ax.plot(x, y, label=f"Ditch {polyline.id}", linewidth=1)

        if log:
            print(f"清沟 {polyline.id}: 包含 {len(polyline.points)} 个点")

    ax.set_title("All Ditches Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()

    ax.set_aspect("equal", adjustable="box")
    plt.grid(True)

    plt.show()


def get_ditch_endpoints(polylines):
    """
    获取所有清沟（ditch）的两个端点。

    :param polylines: 由 load_polylines_from_shp 返回的 Polyline 对象列表。
    :return: 包含每条 Polyline 两个端点的列表，格式为 [(start_point, end_point), ...]。
    """
    endpoints = []

    for idx, polyline in enumerate(polylines):
        if len(polyline.points) >= 2:
            start_point = polyline.points[0]
            end_point = polyline.points[-1]
            endpoints.append((start_point, end_point))
        else:
            print(f"警告：Polyline {polyline.id} 点数不足，无法提取端点。")

    return endpoints

def process_ditch_endpoints(ditchs, closed_shapes,centerline, save_path=None, log=True):

    results = []

    for idx, ditch in enumerate(ditchs):
        if len(ditch.points) < 2:
            print(f"⚠️ 警告：清沟 {ditch.id} 只有一个点，跳过。")
            continue

        start_point = ditch.points[0]
        end_point = ditch.points[-1]

        # 处理起点
        start_index, start_shape = find_point_in_closed_shapes(start_point, closed_shapes)
        if start_shape:
            proj_start_1 = start_shape.tangent_line_1.project(start_point)
            proj_start_2 = start_shape.tangent_line_2.project(start_point)
            proj_start_1_point = start_shape.tangent_line_1.interpolate(proj_start_1)
            proj_start_2_point = start_shape.tangent_line_2.interpolate(proj_start_2)
        else:
            proj_start_1_point = None
            proj_start_2_point = None
            print(f"⚠️ 清沟 {ditch.id} 的起点不在任何 ClosedShape 中。")

        # 处理终点
        end_index, end_shape = find_point_in_closed_shapes(end_point, closed_shapes)
        if end_shape:
            proj_end_1 = end_shape.tangent_line_1.project(end_point)
            proj_end_2 = end_shape.tangent_line_2.project(end_point)
            proj_end_1_point = end_shape.tangent_line_1.interpolate(proj_end_1)
            proj_end_2_point = end_shape.tangent_line_2.interpolate(proj_end_2)
        else:
            proj_end_1_point = None
            proj_end_2_point = None
            print(f"⚠️ 清沟 {ditch.id} 的终点不在任何 ClosedShape 中。")

        # 存储结果
        results.append({
            "ditch_id": ditch.id,
            "start_point": (start_point.x, start_point.y),
            "start_proj_tangent_1": (proj_start_1_point.x, proj_start_1_point.y) if proj_start_1_point else None,
            "start_proj_tangent_2": (proj_start_2_point.x, proj_start_2_point.y) if proj_start_2_point else None,
            "end_point": (end_point.x, end_point.y),
            "end_proj_tangent_1": (proj_end_1_point.x, proj_end_1_point.y) if proj_end_1_point else None,
            "end_proj_tangent_2": (proj_end_2_point.x, proj_end_2_point.y) if proj_end_2_point else None,
        })

        if log:
            fig, ax = plt.subplots(figsize=(12, 8))

            x, y = centerline.line.xy
            ax.plot(x, y, color="gray", linewidth=2)
            # 绘制清沟
            x = [point.x for point in ditch.points]
            y = [point.y for point in ditch.points]
            ax.plot(x, y, label=f"Ditch {ditch.id}", color="blue", linewidth=2)

            # 绘制端点
            ax.scatter(start_point.x, start_point.y, color='red', label='Start Point', zorder=5)
            ax.scatter(end_point.x, end_point.y, color='purple', label='End Point', zorder=5)

            # 绘制起点投影点
            if proj_start_1_point:
                ax.scatter(proj_start_1_point.x, proj_start_1_point.y, color='orange', zorder=5,s=10)
                ax.plot([start_point.x, proj_start_1_point.x], [start_point.y, proj_start_1_point.y], color='orange', linestyle='--')
            if proj_start_2_point:
                ax.scatter(proj_start_2_point.x, proj_start_2_point.y, color='cyan', zorder=5,s=10)
                ax.plot([start_point.x, proj_start_2_point.x], [start_point.y, proj_start_2_point.y], color='cyan', linestyle='--')

            # 绘制终点投影点
            if proj_end_1_point:
                ax.scatter(proj_end_1_point.x, proj_end_1_point.y, color='orange', zorder=5,s=10)
                ax.plot([end_point.x, proj_end_1_point.x], [end_point.y, proj_end_1_point.y], color='orange', linestyle='--')
            if proj_end_2_point:
                ax.scatter(proj_end_2_point.x, proj_end_2_point.y, color='cyan', zorder=5,s=10)
                ax.plot([end_point.x, proj_end_2_point.x], [end_point.y, proj_end_2_point.y], color='cyan', linestyle='--')



            for j, shape in enumerate(closed_shapes):
                hash_input = str(j).encode('utf-8')
                hash_digest = hashlib.md5(hash_input).hexdigest()
                color = '#' + hash_digest[:6]

                px, py = shape.polygon.exterior.xy
                ax.fill(px, py, color=color, alpha=0.5, label=f"Closed Shape {j}" if j == 0 else "")
                ax.plot(px, py, color="red", linewidth=0.7)

            # 设置展示范围：清沟周围上下左右 1000 米
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            max_y = max(y)

            ax.set_xlim(min_x - 5000, max_x + 5000)
            ax.set_ylim(min_y - 5000, max_y + 5000)

            ax.set_title(f"Ditch {ditch.id} with Projections")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True)

            if save_path:
                plt.savefig(f"{save_path}/ditch_{ditch.id}_projections.png", dpi=300, bbox_inches='tight')
                print(f"图像已保存到 {save_path}/ditch_{ditch.id}_projections.png")
                plt.close()
            else:
                plt.show()
    return results


def main():

    # 中心线
    centerline_file = "D:/机器学习数据/中心线和南北岸线/中心线平滑.shp"
    polylines = load_polylines_from_shp(centerline_file, False)
    merged_center_line = merge_polylines(polylines, False)

    # 边界线
    boundary_file = "D:/机器学习数据/中心线和南北岸线/南北线_修改后.shp"
    work_polylines = load_polylines_from_shp(boundary_file, False)

    north_south_file= "north_south_line.json"

    if os.path.exists(north_south_file):
        print(f"Found existing closed shapes file: {north_south_file}. Loading closed shapes...")
        north_line,south_line=load_north_south_lines_from_json(north_south_file)

    else :
        work_polyline=work_polylines[0].line
        coords = list(work_polyline.coords)
        min_x_point = min(coords, key=lambda p: p[0])
        min_y_point = min(coords, key=lambda p: p[1])
        min_x_point = Point(min_x_point)
        min_y_point = Point(min_y_point)
        min_x_index = coords.index(min_x_point.coords[0])
        min_y_index = coords.index(min_y_point.coords[0])

        north_line,south_line = split_polyline_by_points(work_polyline,min_x_point,min_y_point,min_x_index,min_y_index)
        save_north_south_lines_to_json(north_line, south_line, "north_south_line.json")

    # 分割点结果文件
    split_points_file = "split_points.yaml"
    if os.path.exists(split_points_file):
        print(f"Found existing split points file: {split_points_file}. Loading split points...")
        split_points = load_split_points_from_file(split_points_file, is_yaml=True)

    else:
        print("No existing split points file found. Running calculations...")
        # plot_normals_and_lines(merged_center_line.line, north_line, south_line, interval=1000)
        split_points = generate_infinite_normals_on_linestring_with_polyline(merged_center_line.line, north_line,south_line, interval=1000)
        save_split_points_to_file(split_points, split_points_file, file_format="yaml")
        # print(f"Split points saved to {split_points_file}")


    # Closed Shapes 结果文件
    closed_shapes_file = "closed_shapes.yaml"
    if os.path.exists(closed_shapes_file):
        print(f"Found existing closed shapes file: {closed_shapes_file}. Loading closed shapes...")
        closed_shapes = load_closed_shapes_from_file(closed_shapes_file, is_yaml=True)
    else:
        print("No existing closed shapes file found. Generating closed shapes...")
        closed_shapes = plot_closed_shapes_with_polylines(split_points,north_line,south_line, save="D:\\code\\shpdealer\\result2")
        save_closed_shapes_to_file(closed_shapes, closed_shapes_file, file_format="yaml")

    # 清沟
    ditch_file="D:\\机器学习数据\\河道中心线和清沟样例\\河道中心线和清沟样例\\20230305清沟_hz.shp"
    ditchs = load_polylines_from_shp(ditch_file, False)
    process_ditch_endpoints(ditchs,closed_shapes,merged_center_line,r"D:\code\shpdealer\result3",True)


if __name__ == "__main__":
    main()
