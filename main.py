import typer
import os
from convert_crs import convert_crs
from crop_xy import crop_file
from image_intersection import image_intersection
from crop import shapefile_to_bw_label
from img2tif import convert_img_to_tiff
from resizer import resize
from imggetter import process_file
from sample_area_getter import extract_polygon
from masker import apply_mask
from auto_sample_area import create_concave_hull_polygon
from split_image import split_images

app = typer.Typer()

os.environ['PROJ_LIB'] = r'D:\code\shpdealer\venv\Lib\site-packages\pyproj\proj_dir\share\proj'

app.command(name="apply-mask", help="将感兴趣区域(AOI)外的图像部分置为0")(apply_mask)
app.command(name="combine-polygons", help="将给定的Shapefile中的所有标注区域合并为一个大的样本区，并根据坐标限制范围")(create_concave_hull_polygon)
app.command(name="convert-crs", help="切换参考系")(convert_crs)
app.command(name="crop-by-coordinates", help="给出上下左右坐标进行裁剪")(crop_file)
app.command(name="crop-label-from-shapefile", help="根据shapefile生成黑白label图像")(shapefile_to_bw_label)
app.command(name="extract-polygon-coordinates", help="从 Shapefile 中提取多边形坐标并保存为 NumPy 文件")(extract_polygon)
app.command(name="img-to-tiff", help="将img图像转化为tif图像")(convert_img_to_tiff)
app.command(name="image-info", help="根据文件路径自动判断文件类型（影像、图片或 Shapefile）并获取基本信息")(process_file)
app.command(name="resize-image", help="压缩图片大小")(resize)
app.command(name="split-image", help="切割图片")(split_images)

if __name__ == "__main__":
    app()
