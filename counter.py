# main_cn_interval_fixed_with_inline_logs.py
import os
import typer
from PIL import Image
from collections import Counter
import concurrent.futures
import logging


# --- 用于 Typer 的自定义内联日志设置 ---

class TyperLogHandler(logging.Handler):
    """一个使用 typer.secho 进行输出的日志处理器，以确保与进度条等UI元素兼容。"""

    def emit(self, record: logging.LogRecord) -> None:
        color = None
        if record.levelno >= logging.ERROR:
            color = typer.colors.RED
        elif record.levelno >= logging.WARNING:
            color = typer.colors.YELLOW

        # 使用 err=True 将日志输出到标准错误流，这有助于避免与标准输出流（如管道）混淆
        typer.secho(self.format(record), fg=color, err=True)


# 1. 获取一个自定义的 logger 实例
logger = logging.getLogger("color_analyzer")

# 2. 设置日志级别
logger.setLevel(logging.INFO)

# 3. 创建一个格式化器
# 对于内联输出，使用一个更简洁的格式
formatter = logging.Formatter('%(levelname)s: [%(threadName)s] %(message)s')

# 4. 创建并设置我们的自定义处理器
handler = TyperLogHandler()
handler.setFormatter(formatter)

# 5. 将处理器添加到 logger (仅在尚未存在时，以防重复添加)
if not logger.handlers:
    logger.addHandler(handler)
    logger.propagate = False  # 防止日志消息被传递到根logger


# --- 日志设置结束 ---


def _process_single_image_for_color_analysis(file_path: str) -> tuple[Counter, int]:
    """
    处理单张图片，进行颜色统计。
    此版本经过优化，避免一次性将所有像素加载到内存中。
    """
    logger.info(f"开始处理: {os.path.basename(file_path)}")
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            pixels_iterator = img_rgb.getdata()
            num_pixels = img_rgb.width * img_rgb.height
            color_counter = Counter(pixels_iterator)
            logger.info(f"处理完成: {os.path.basename(file_path)}")
            return color_counter, num_pixels
    except Exception as e:
        # 使用 logger 来报告错误，它会自动使用 TyperLogHandler 进行格式化和彩色输出
        logger.error(f"处理图片 '{os.path.basename(file_path)}' 时出错: {e}",
                     exc_info=False)  # exc_info=False 避免重复打印异常类
        return Counter(), 0


def _print_color_report(
        color_counter: Counter,
        top_n: int,
        processed_count: int,
        total_images: int,
        is_final: bool = False,
):
    """打印格式化的颜色统计报告。"""
    if not color_counter:
        logger.warning("尝试打印报告，但颜色计数器为空。")
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
        target_folder: str = typer.Option(
            None, "--target_folder", "-t", help="包含待分析图片的文件夹路径。",
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
    """
    分析指定文件夹中所有图片的颜色构成，并统计最常见的颜色。
    此版本优化了内存使用，可以处理大量或高分辨率的图片。
    """
    logger.info("颜色分析程序启动。")
    if not target_folder or not os.path.isdir(target_folder):
        logger.error("错误：必须提供一个有效的目标文件夹路径。", exc_info=False)
        raise typer.Exit(code=1)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    try:
        image_files = [
            os.path.join(target_folder, f)
            for f in os.listdir(target_folder)
            if os.path.isfile(os.path.join(target_folder, f)) and os.path.splitext(f)[1].lower() in image_extensions
        ]
    except FileNotFoundError:
        logger.error(f"错误：找不到文件夹 '{target_folder}'。", exc_info=False)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"读取文件夹时出错: {e}", exc_info=True)
        raise typer.Exit(code=1)

    total_images = len(image_files)
    if not total_images:
        logger.warning("在指定文件夹中未找到支持的图片文件。")
        raise typer.Exit()

    logger.info(f"找到 {total_images} 张图片。开始使用最多 {max_workers} 个线程进行分析...")

    color_counter = Counter()
    total_pixels = 0
    processed_count = 0

    with typer.progressbar(length=total_images, label="正在分析图片") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_image_for_color_analysis, file_path): file_path for file_path in
                       image_files}

            for future in concurrent.futures.as_completed(futures):
                try:
                    counter, pixel_count = future.result()
                    if pixel_count > 0:  # 仅在成功处理后更新
                        color_counter.update(counter)
                        total_pixels += pixel_count
                except Exception as exc:
                    file_path = futures[future]
                    # 当 future.result() 抛出异常时，在此处记录详细信息
                    logger.error(f"处理文件 '{os.path.basename(file_path)}' 时线程内产生了一个未捕获的异常: {exc}",
                                 exc_info=True)

                processed_count += 1
                progress.update(1)

                if interval > 0 and processed_count % interval == 0 and processed_count < total_images:
                    logger.info(f"已处理 {processed_count} 张图片，生成中期报告。")
                    _print_color_report(color_counter, top_n, processed_count, total_images, is_final=False)

    if processed_count > 0 and len(color_counter) > 0:
        logger.info(f"分析完成。总计处理了 {total_pixels:,} 个像素。")
        _print_color_report(color_counter, top_n, processed_count, total_images, is_final=True)
        footer = typer.style("\n--- 分析结束 ---", fg=typer.colors.BRIGHT_GREEN, bold=True)
        typer.echo(footer)
    else:
        logger.error("未能成功处理任何图片，请检查文件夹内容、文件权限或图片格式。")


if __name__ == "__main__":
    typer.run(analyze_colors)