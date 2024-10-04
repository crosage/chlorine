import os
import argparse
from PIL import Image
import typer

def resize(
    dataset_path: str = typer.Option(..., "--dataset-path", "-d", help="图像数据集的路径。"),
    width: int = typer.Option(..., "--width", "-w", help="要调整图像的宽度。"),
    height: int = typer.Option(..., "--height", "-h", help="要调整图像的高度。"),
    filter_type: str = typer.Option('lanczos', "--filter", "-f", help="用于调整大小的过滤器类型。", case_sensitive=False)
):
    if filter_type not in ['lanczos', 'nearest', 'bilinear', 'bicubic']:
        typer.echo(f"Invalid filter type: {filter_type}")
        raise typer.Exit()

    if not os.path.exists(dataset_path):
        print(f"The dataset path {dataset_path} does not exist.")
        return

    output_path = os.path.join(dataset_path, 'resized')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 处理目录中的每个文件
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(dataset_path, filename)
            try:
                # 打开图像
                with Image.open(img_path) as img:
                    # 调整图像大小
                    img_resized = img.resize((width, height), filter_type)
                    # 保存图像到输出目录
                    img_resized.save(os.path.join(output_path, filename))
                    print(f"Resized and saved {filename} to {output_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
