import os
from PIL import Image
import typer
import numpy as np


def convert_to_grayscale(
        input_dir: str = typer.Option(..., '-i', '--input-dir', help="输入图像文件的文件夹路径"),
        overwrite: bool = typer.Option(True, '-o', '--overwrite', help="是否覆盖原文件，默认为True")
):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(input_dir, filename)
            image = Image.open(file_path).convert("RGB")

            image_np = np.array(image)

            gray_image_np = image_np[:, :, 0]

            gray_image = Image.fromarray(gray_image_np, mode='L')

            if overwrite:
                gray_image.save(file_path)
                typer.echo(f"Processed and overwritten: {file_path}")
            else:
                new_file_path = os.path.join(input_dir, f"gray_{filename}")
                gray_image.save(new_file_path)
                typer.echo(f"Processed and saved as: {new_file_path}")
