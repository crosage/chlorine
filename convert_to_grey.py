import os
import json
import typer
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器
import sys


def process_image(args):

    filename, input_dir, palette, overwrite = args
    file_path = os.path.join(input_dir, filename)

    try:
        image = Image.open(file_path).convert("RGB")
        image_np = np.array(image)
        height, width, _ = image_np.shape

        gray_image_np = np.zeros((height, width), dtype=np.uint8)

        for class_index, color in enumerate(palette):
            mask = np.all(image_np == np.array(color), axis=-1)
            gray_image_np[mask] = class_index

        gray_image = Image.fromarray(gray_image_np, mode='L')

        if overwrite:
            output_path = file_path
            gray_image.save(output_path)
        else:
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_gray.png"
            output_path = os.path.join(input_dir, new_filename)
            gray_image.save(output_path)
        return True

    except Exception as e:
        tqdm.write(f"\n处理文件 {filename} 时出错: {e}", file=sys.stderr)
        return False


def convert_palette_to_grayscale(
        input_dir: str = typer.Option(
            ..., '-i', '--input-dir',
            help="包含源RGB图像的文件夹路径。"
        ),
        palette_path: str = typer.Option(
            ..., '-p', '--palette',
            help="定义颜色到类别索引映射的JSON文件路径。"
        ),
        overwrite: bool = typer.Option(
            False, '--overwrite',
            help="是否覆盖原文件。如果为False，则会创建带'gray_'前缀的新文件。"
        ),
        workers: int = typer.Option(
            os.cpu_count(), '-w', '--workers',  # 新增参数，用于控制线程数
            help="用于并行处理的工作线程数量，默认为CPU核心数。"
        )
):
    try:
        with open(palette_path, 'r') as f:
            palette = json.load(f)
        typer.secho(f"成功从 '{palette_path}' 加载调色板。", fg=typer.colors.GREEN)
    except FileNotFoundError:
        typer.secho(f"错误: 调色板文件未找到 '{palette_path}'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        typer.secho(f"错误: 调色板文件 '{palette_path}' 不是一个有效的JSON格式。", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        typer.secho(f"在目录 '{input_dir}' 中未找到支持的图像文件。", fg=typer.colors.YELLOW)
        raise typer.Exit()

    typer.echo(f"找到 {len(image_files)} 个图像文件。将使用 {workers} 个线程开始并行转换...")


    tasks = [(filename, input_dir, palette, overwrite) for filename in image_files]

    with ThreadPoolExecutor(max_workers=workers) as executor:

        results = list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="转换进度"))

    success_count = sum(1 for r in results if r)
    typer.secho(f"\n转换完成！成功处理 {success_count}/{len(image_files)} 个文件。", fg=typer.colors.BRIGHT_GREEN)


if __name__ == "__main__":
    typer.run(convert_palette_to_grayscale)