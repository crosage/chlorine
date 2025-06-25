# main_cn_interval.py
import os
import typer
from PIL import Image
from collections import Counter
import concurrent.futures

def _process_single_image_for_color_analysis(file_path: str) -> tuple[Counter, int]:
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            pixels = list(img_rgb.getdata())
            return Counter(pixels), len(pixels)
    except Exception as e:
        typer.secho(f"处理图片 '{os.path.basename(file_path)}' 时出错: {e}", fg=typer.colors.RED, err=True)
        return Counter(), 0


def _print_color_report(
        color_counter: Counter,
        top_n: int,
        processed_count: int,
        total_images: int,
        is_final: bool = False,
):
    if not color_counter:
        return

    if is_final:
        header_text = f"\n--- 最终统计结果 (共处理 {processed_count} 张图片) ---"
        header_color = typer.colors.BRIGHT_GREEN
    else:
        header_text = f"\n--- 中期统计报告 (已处理 {processed_count}/{total_images} 张图片) ---"
        header_color = typer.colors.YELLOW

    header = typer.style(header_text, fg=header_color, bold=True)
    typer.echo(header)

    sorted_colors = color_counter.most_common()
    results_to_show = sorted_colors[:top_n] if top_n > 0 else sorted_colors

    if top_n > 0 and len(sorted_colors) > top_n:
        typer.echo(f"(共 {len(sorted_colors):,} 种颜色，仅显示前 {top_n} 种)")

    typer.echo(f"\n{'排名':<5} | {'RGB 值':<20} | {'出现次数'}")
    typer.echo("-" * 45)

    for i, (color, count) in enumerate(results_to_show):
        rank = f"{i + 1:<5}"
        rgb_val = f"({color[0]:>3}, {color[1]:>3}, {color[2]:>3})"
        count_val = f"{count:,}"
        rgb_styled = typer.style(rgb_val, fg=typer.colors.CYAN)
        typer.echo(f"{rank} | {rgb_styled:<20} | {count_val}")


def analyze_colors(
        target_folder: str = typer.Argument(
            ...,
            help="包含待分析图片的文件夹路径。",
            exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True,
        ),
        max_workers: int = typer.Option(
            12, "--workers", "-w", help="用于处理图像的最大工作线程数。",
        ),
        top_n: int = typer.Option(
            20, "--top", "-n", help="仅显示最常见的前 N 种颜色。设置为 0 则显示所有颜色。",
        ),
        interval: int = typer.Option(
            0, "--interval", "-i", help="每处理 N 张图片后进行一次中期统计。设置为 0 则仅在最后统计。", min=0,
        ),
):

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    image_files = [
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder)
        if os.path.isfile(os.path.join(target_folder, f)) and os.path.splitext(f)[1].lower() in image_extensions
    ]

    total_images = len(image_files)
    if not total_images:
        typer.echo("在指定文件夹中未找到支持的图片文件。")
        raise typer.Exit()

    typer.echo(f"找到 {total_images} 张图片。开始多线程分析...")

    color_counter = Counter()
    total_pixels = 0
    processed_count = 0

    with typer.progressbar(total=total_images, label="正在分析图片") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single_image_for_color_analysis, file_path) for file_path in
                       image_files]

            for future in concurrent.futures.as_completed(futures):
                counter, pixel_count = future.result()
                color_counter.update(counter)
                total_pixels += pixel_count
                processed_count += 1
                progress.update(1)

                if interval > 0 and processed_count % interval == 0 and processed_count < total_images:
                    _print_color_report(color_counter, top_n, processed_count, total_images, is_final=False)

    if processed_count > 0:
        typer.echo(f"\n分析完成。总计处理了 {total_pixels:,} 个像素。")
        _print_color_report(color_counter, top_n, processed_count, total_images, is_final=True)
        footer = typer.style("\n--- 分析结束 ---", fg=typer.colors.BRIGHT_GREEN, bold=True)
        typer.echo(footer)
    else:
        typer.secho("未能完成颜色分析，请检查文件夹内容或文件权限。", fg=typer.colors.RED, bold=True)

if __name__=="__main__":
    app = typer.Typer()
    os.environ['PROJ_LIB'] = r'D:\code\shpdealer\venv\Lib\site-packages\pyproj\proj_dir\share\proj'
    app.command(name="analyze-colors", help="分析一个文件夹中所有图片的颜色频率，并支持中期报告")(analyze_colors)
    app()