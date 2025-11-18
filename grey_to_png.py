import os
import json
import typer
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器
import sys


def process_grayscale_to_rgb(args):
    filename, input_dir, output_dir, palette_array, overwrite = args
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    if not overwrite and os.path.exists(output_path):
        return False

    try:
        img = Image.open(input_path)
        if img.mode not in ['L', 'LA']:
            print(f"信息: 文件 {filename} 不是灰度图 (模式: {img.mode})，已跳过。")
            return False
        if img.mode == 'LA':
            img = img.convert('L')

        gray_array = np.array(img)
        if gray_array.dtype != np.uint8:
            print(f"诊断信息: 文件 {filename} 的数据类型是 {gray_array.dtype}，将强制转换为 uint8。")
            gray_array = gray_array.astype(np.uint8)
        rgb_array = palette_array[gray_array]
        new_img = Image.fromarray(rgb_array.astype(np.uint8), 'RGB')
        new_img.save(output_path)
        return True

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")
        return False

def convert_grayscale_to_palette(
        input_dir: str = typer.Option(
            ..., '-i', '--input-dir',
            help="包含源R图像的文件夹路径。"
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
            os.cpu_count(), '-w', '--workers',
            help="用于并行处理的工作线程数量，默认为CPU核心数。"
        )
):
    try:
        with open(palette_path, 'r') as f:
            palette_list = json.load(f) # 变量名改为 palette_list 以示区分
        typer.secho(f"成功从 '{palette_path}' 加载调色板。", fg=typer.colors.GREEN)
    except FileNotFoundError:
        typer.secho(f"错误: 调色板文件未找到 '{palette_path}'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        typer.secho(f"错误: 调色板文件 '{palette_path}' 不是一个有效的JSON格式。", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        palette_array = np.array(palette_list, dtype=np.uint8)
        typer.echo(f"调色板已转换为Numpy数组，形状: {palette_array.shape}, 类型: {palette_array.dtype}")
    except Exception as e:
        typer.secho(f"错误: 无法将调色板转换为Numpy数组: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if palette_array.ndim != 2 or palette_array.shape[1] != 3:
        typer.secho(f"错误: 调色板数组的形状应为(N, 3)，但当前为{palette_array.shape}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        typer.secho(f"在目录 '{input_dir}' 中未找到支持的图像文件。", fg=typer.colors.YELLOW)
        raise typer.Exit()

    typer.echo(f"找到 {len(image_files)} 个图像文件。将使用 {workers} 个线程开始并行转换...")

    tasks = [(filename, input_dir, input_dir, palette_array, overwrite) for filename in image_files]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_grayscale_to_rgb, tasks), total=len(tasks), desc="转换进度"))

    success_count = sum(1 for r in results if r)
    typer.secho(f"\n转换完成！成功处理 {success_count}/{len(image_files)} 个文件。", fg=typer.colors.BRIGHT_GREEN)


if __name__ == "__main__":
    typer.run(convert_grayscale_to_palette)