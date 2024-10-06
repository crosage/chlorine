import typer
import os
from image_intersection import image_intersection
from crop import crop_within_sample_area
from img2tif import convert_img_to_tiff
from resizer import resize
from imggetter import process_file
from sample_area_getter import extract_polygon
from masker import apply_mask
from auto_sample_area import create_concave_hull_polygon
app = typer.Typer()

os.environ['PROJ_LIB'] = r'D:\code\shpdealer\venv\Lib\site-packages\pyproj\proj_dir\share\proj'

app.command(name="crop",help="根据shapefile生成黑白label图像")(crop_within_sample_area)
app.command(name="resize",help="压缩图片大小")(resize)
app.command(name="image-info", help="根据文件路径自动判断文件类型（影像、图片或 Shapefile）并获取基本信息")(process_file)
app.command(name="extract-polygon", help="从 Shapefile 中提取多边形坐标并保存为 NumPy 文件")(extract_polygon)
app.command(name="mask-outside", help="将感兴趣区域(AOI)外的图像部分置为0")(apply_mask)
app.command(name="combine-polygon", help="将给定的Shapefile中的所有标注区域合并为一个大的样本区，并根据坐标限制范围")(create_concave_hull_polygon)
app.command(name="img2tif",help="将img图像转化为tif图像")(convert_img_to_tiff)
app.command(name="convert-and-crop",help="给出上下左右坐标进行裁剪")(image_intersection)

if __name__ == "__main__":
    app()
