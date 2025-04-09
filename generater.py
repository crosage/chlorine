import os
import typer
from PIL import Image
import rasterio # <-- Import Rasterio
from pathlib import Path # <-- Use Pathlib for better path handling
# typing_extensions Annotated is not used here, so removed unless needed elsewhere

# Define the Typer app (good practice if you might add more commands later)
app = typer.Typer()

@app.command()
def generate_dummy_ground_truth(
    input_dir: Path = typer.Option(
        ..., '-i', '--input-dir',
        help="包含原始图片的文件夹路径",
        exists=True, # Ensure input directory exists
        file_okay=False, # It must be a directory
        dir_okay=True,
        readable=True, # Check read permissions
        resolve_path=True # Convert to absolute path
    ),
    output_dir: Path = typer.Option(
        ..., '-o', '--output-dir',
        help="保存假设真值图片的文件夹路径",
        file_okay=False, # Must be a directory path
        dir_okay=True,
        writable=True, # Check write permissions (implicitly checks parent exists)
        resolve_path=True
    ),
    output_extension: str = typer.Option(
        ".png", '-e', '--extension',
        help="保存的假设真值图片的扩展名 (例如 .png, .tif)"
    ),
):
    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Create output dir if needed
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Output extension: {output_extension}")

        processed_count = 0
        skipped_count = 0
        error_count = 0
        for item_path in input_dir.iterdir():
            if not item_path.is_file():
                continue

            filename = item_path.name
            lower_filename = filename.lower()
            width, height = None, None

            print(f"Processing: {filename}...")

            try:
                if lower_filename.endswith(('.tif', '.tiff')):
                    try:
                        with rasterio.open(item_path) as src:
                            width = src.width
                            height = src.height
                        print(f"  Read dimensions via Rasterio ({width}x{height})")
                    except rasterio.RasterioIOError as rio_err:
                        print(f"  Error: Rasterio failed to read {filename}: {rio_err}. Skipping.")
                        error_count += 1
                        continue
                elif lower_filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        with Image.open(item_path) as img:
                            width, height = img.size
                        print(f"  Read dimensions via Pillow ({width}x{height})")
                    except Exception as pil_err:
                        print(f"  Error: Pillow failed to read {filename}: {pil_err}. Skipping.")
                        error_count += 1
                        continue
                else:
                    print(f"  Skipped: Unsupported file extension for {filename}")
                    skipped_count += 1
                    continue
                if width is not None and height is not None and width > 0 and height > 0:
                    dummy_gt = Image.new("L", (width, height), 0)
                    base_name = item_path.stem
                    if not output_extension.startswith('.'):
                        output_extension = '.' + output_extension
                    output_filename = f"{base_name}{output_extension}"
                    output_path = output_dir / output_filename
                    dummy_gt.save(output_path)
                    print(f"  -> Saved dummy GT: {output_filename}")
                    processed_count += 1
                else:
                    print(f"  Skipped: Invalid dimensions obtained for {filename} ({width}x{height})")
                    skipped_count += 1


            except Exception as e:
                print(f"  Error: Unexpected problem processing {filename}: {e}")
                error_count += 1

        print("\n--- Generation Summary ---")
        print(f"Successfully processed and generated GT for: {processed_count} files")
        print(f"Skipped (unsupported format or invalid dimensions): {skipped_count} files")
        print(f"Encountered errors during reading: {error_count} files")
        print("Dummy ground truth generation complete!")

    except Exception as general_err:
        print(f"An overall error occurred: {general_err}")


if __name__ == "__main__":
    app() # Run the Typer app