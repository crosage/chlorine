import cv2
import math
import numpy as np
import sys
import os
from pathlib import Path
import rasterio
import rasterio.windows # Explicit import for clarity
import tifffile
from tqdm import tqdm
import warnings
from typing import Tuple, Optional

import typer # Import typer

# --- Constants ---
NUMPY_EPS = np.finfo(np.float64).eps
UINT16_MAX = 65535.0

# --- Helper Functions ---

def check_input_image(im: np.ndarray, expected_channels: int, func_name: str) -> None:
    """Validates the input image dimensions and number of channels."""
    if im.ndim != 3 or im.shape[2] != expected_channels:
        raise ValueError(
            f"{func_name}: 输入图像需要是 HxWx{expected_channels} 格式, 收到 {im.shape}"
        )
    if im.dtype != np.float64:
        # This warning is less critical now as conversion happens anyway
        # warnings.warn(
        #     f"{func_name}: 输入图像数据类型为 {im.dtype}, 期望 float64. "
        #     "内部计算将使用 float64。",
        #     UserWarning
        # )
        pass # Silently handle conversion

# --- Core Dehazing Functions (Now accept parameters) ---

def dark_channel(im: np.ndarray, sz: int) -> np.ndarray:
    """
    计算 N 通道图像的暗通道图。

    Args:
        im: 输入图像 (HxWxN, float64, 范围 [0, 1])。
        sz: 计算暗通道时使用的邻域（patch）的大小（边长）。

    Returns:
        暗通道图 (HxW, float64)。
    """
    n_channels = im.shape[2]
    # check_input_image(im, n_channels, "dark_channel") # Called internally now

    dc = np.min(im, axis=2)

    # Ensure kernel size is odd and positive
    sz = max(1, int(sz))
    if sz % 2 == 0:
        sz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def atmospheric_light(im: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    根据暗通道图估计 N 通道图像的全局大气光 A。

    Args:
        im: 原始输入图像 (HxWxN, float64, 范围 [0, 1])。
        dark: 暗通道图 (HxW, float64)。

    Returns:
        估计的大气光值 A (1xN, float64)。
    """
    n_channels = im.shape[2]
    # check_input_image(im, n_channels, "atmospheric_light") # Called internally now
    if dark.ndim != 2 or dark.shape[:2] != im.shape[:2]:
         raise ValueError(f"atmospheric_light: 暗通道图尺寸 {dark.shape} 与图像尺寸 {im.shape[:2]} 不匹配")

    h, w = im.shape[:2]
    imsz = h * w
    num_pixels_to_consider = int(max(math.floor(imsz / 1000), 1)) # Top 0.1%

    dark_flat = dark.reshape(imsz)
    im_flat = im.reshape(imsz, n_channels)

    # Indices of the brightest pixels in the dark channel
    brightest_indices = np.argsort(dark_flat)[-num_pixels_to_consider:]

    # Average the corresponding pixels in the original image
    A = np.mean(im_flat[brightest_indices], axis=0, keepdims=True)

    return A

def transmission_estimate(im: np.ndarray, A: np.ndarray, sz: int, omega: float) -> np.ndarray:
    """
    初步估计 N 通道图像的透射率图。

    Args:
        im: 原始输入图像 (HxWxN, float64, 范围 [0, 1])。
        A: 估计的大气光 (1xN, float64)。
        sz: 计算暗通道时使用的邻域大小。
        omega: 保留少量雾的系数。

    Returns:
        初步估计的透射率图 (HxW, float64)。
    """
    n_channels = im.shape[2]
    # check_input_image(im, n_channels, "transmission_estimate") # Called internally now
    if A.shape != (1, n_channels):
        raise ValueError(f"transmission_estimate: 大气光 A 形状错误. A:{A.shape}, 需要 (1, {n_channels})")

    # Ensure A is not zero for division
    A_clipped = np.maximum(A, NUMPY_EPS)
    im_normalized = np.empty_like(im)
    for i in range(n_channels):
        im_normalized[:, :, i] = im[:, :, i] / A_clipped[0, i]

    # Calculate transmission using the dark channel of the normalized image
    transmission = 1.0 - omega * dark_channel(im_normalized, sz)
    return transmission

def guided_filter(im_guide: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    引导滤波实现。

    Args:
        im_guide: 引导图像 (HxW 或 HxWxC, float64)。建议使用灰度图。
        p: 需要滤波的输入图像 (HxW, float64)。
        r: 滤波器的窗口半径。
        eps: 正则化参数。

    Returns:
        滤波后的图像 (HxW, float64)。
    """
    # Ensure float64 for calculations
    if im_guide.dtype != np.float64: im_guide = im_guide.astype(np.float64)
    if p.dtype != np.float64: p = p.astype(np.float64)

    # If guide is color, use its grayscale version (average)
    if im_guide.ndim == 3:
        # warnings.warn("guided_filter: 引导图像是多通道，将使用其平均值进行滤波。", UserWarning)
        guide = np.mean(im_guide, axis=2)
    elif im_guide.ndim == 2:
        guide = im_guide
    else:
        raise ValueError(f"guided_filter: 不支持的引导图像维度 {im_guide.ndim}")

    # Ensure radius is positive for box filter kernel size
    r_kernel = max(1, int(r))
    ksize = (r_kernel, r_kernel)

    mean_I = cv2.boxFilter(guide, cv2.CV_64F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_64F, ksize)
    mean_Ip = cv2.boxFilter(guide * p, cv2.CV_64F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p # Eq(6) in the paper; covariance p = aI + b => cov(p, I) = a * var(I)

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, ksize)
    var_I = mean_II - mean_I * mean_I # Eq(7) in the paper; variance

    # Calculate 'a' and 'b' (Eq(8) in the paper)
    # eps prevents division by zero when var_I is small
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I # Eq(9)

    # Average 'a' and 'b' over the window (Eq(13), Eq(14))
    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

    # Calculate final filtered output (Eq(1))
    q = mean_a * guide + mean_b
    return q

def transmission_refine(im_float_bgrn: np.ndarray, et: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    使用引导滤波优化透射率图。使用 BGR 通道转化的灰度图作为引导。

    Args:
        im_float_bgrn: 原始输入图像 (HxWx4, float64, 范围 [0, 1])。
        et: 初步估计的透射率图 (HxW, float64)。
        r: 引导滤波半径。
        eps: 引导滤波正则化参数。

    Returns:
        优化后的透射率图 (HxW, float64)。
    """
    # check_input_image(im_float_bgrn, 4, "transmission_refine") # Called internally now

    # Use BGR channels to create the grayscale guide image
    im_float_bgr = im_float_bgrn[:, :, :3]
    # Convert to float32 for cvtColor if necessary (though guided_filter converts guide to float64 anyway)
    if im_float_bgr.dtype != np.float32:
        im_float_bgr = im_float_bgr.astype(np.float32)

    gray_guide = cv2.cvtColor(im_float_bgr, cv2.COLOR_BGR2GRAY)
    # gray_guide is HxW, float32

    # Refine the initial transmission estimate 'et' using the grayscale guide
    t = guided_filter(gray_guide, et, r, eps)
    return t

def recover(im: np.ndarray, t: np.ndarray, A: np.ndarray, tx: float) -> np.ndarray:
    """
    根据大气散射模型恢复 N 通道无雾图像。 J = (I - A) / max(t, tx) + A

    Args:
        im: 原始输入图像 (HxWxN, float64, 范围 [0, 1])。
        t: 优化后的透射率图 (HxW, float64)。
        A: 估计的大气光 (1xN, float64)。
        tx: 透射率的下限阈值。

    Returns:
        恢复的无雾图像 (HxWxN, float64)，值被裁剪到 [0, 1]。
    """
    n_channels = im.shape[2]
    # check_input_image(im, n_channels, "recover") # Called internally now
    if A.shape != (1, n_channels):
        raise ValueError(f"recover: 大气光 A 形状错误. A:{A.shape}, 需要 (1, {n_channels})")
    if t.ndim != 2 or t.shape[:2] != im.shape[:2]:
        raise ValueError(f"recover: 透射率图 t 尺寸 {t.shape} 与图像尺寸 {im.shape[:2]} 不匹配")

    # Clamp transmission map to avoid division by zero or near-zero
    t_clipped = np.maximum(t, tx) # max(t, tx)

    # Expand transmission map to match image channels for broadcasting
    t_expanded = np.expand_dims(t_clipped, axis=2) # HxWx1

    # Apply the recovery formula channel-wise
    res = (im - A) / t_expanded + A

    # Clip result to valid range [0, 1]
    return np.clip(res, 0, 1)

# --- File Handling and Tiling ---

def estimate_global_A(filepath: Path, scale_factor: int, dark_patch_sz: int) -> Optional[np.ndarray]:
    """
    从低分辨率图像估算全局大气光 A。

    Args:
        filepath: 输入图像文件路径 (TIFF)。
        scale_factor: 下采样因子。
        dark_patch_sz: 计算暗通道时使用的 patch 大小。

    Returns:
        估算的全局大气光 A (1x4)，如果出错则返回 None。
    """
    typer.echo(f"开始估算全局大气光 A (下采样因子: {scale_factor}, 暗通道patch: {dark_patch_sz})...")
    try:
        with tifffile.TiffFile(filepath) as tif:
            if not tif.series or len(tif.series[0].shape) < 3:
                raise ValueError(f"TIFF 文件 {filepath} 无效或缺少有效的 Series/Page 或维度不足。")

            original_shape = tif.series[0].shape
            original_dtype = tif.series[0].dtype
            num_channels = original_shape[2]

            if num_channels != 4:
                raise ValueError(f"图像通道数 ({num_channels}) 不是 4")

            H, W = original_shape[:2]
            new_H, new_W = H // scale_factor, W // scale_factor

             # Ensure patch size isn't larger than downscaled image
            if new_H < dark_patch_sz or new_W < dark_patch_sz:
                typer.echo(f"警告：下采样后尺寸 ({new_H}, {new_W}) 小于暗通道patch尺寸 ({dark_patch_sz})。")
                # Option 1: Reduce patch size for this estimation
                adjusted_patch_sz = min(dark_patch_sz, new_H, new_W)
                if adjusted_patch_sz % 2 == 0: adjusted_patch_sz -=1
                adjusted_patch_sz = max(1, adjusted_patch_sz)
                typer.echo(f"将在此步骤中使用调整后的暗通道patch尺寸: {adjusted_patch_sz}")
                dark_patch_sz = adjusted_patch_sz # Use adjusted size locally
                # Option 2 (Original behavior): Try full resolution (memory intensive)
                # typer.echo("将尝试使用原始图像估算 A（可能耗尽内存）。")
                # scale_factor = 1
                # new_H, new_W = H, W
                # try:
                #     img_low_res_uint = tif.series[0].asarray()
                # except MemoryError:
                #      typer.echo("错误：直接读取完整图像以估算 A 时内存不足。无法估算全局 A。", err=True)
                #      return None
                # except Exception as e:
                #     typer.echo(f"直接读取完整图像时出错: {e}", err=True)
                #     return None

            # Proceed with downsampling (or use full res if fallback was chosen above)
            if scale_factor > 1:
                typer.echo(f"  读取并下采样图像到 ({new_H}, {new_W})...")
                try:
                    img_full_res_uint = tif.series[0].asarray()
                    img_low_res_uint = cv2.resize(img_full_res_uint, (new_W, new_H), interpolation=cv2.INTER_AREA)
                    del img_full_res_uint
                except MemoryError:
                    typer.echo("错误：读取完整图像以进行下采样时内存不足。无法估算全局 A。", err=True)
                    typer.echo("建议：尝试更大的 scale_factor 或确保有足够内存。", err=True)
                    return None
                except Exception as e:
                    typer.echo(f"读取或调整图像大小时出错: {e}", err=True)
                    return None
            else: # scale_factor is 1 (either originally or fallback)
                try:
                    img_low_res_uint = tif.series[0].asarray()
                except MemoryError:
                     typer.echo("错误：读取完整图像以估算 A 时内存不足。无法估算全局 A。", err=True)
                     return None
                except Exception as e:
                    typer.echo(f"读取完整图像时出错: {e}", err=True)
                    return None


            # Convert to float64 [0, 1]
            if img_low_res_uint.dtype != np.uint16:
                warnings.warn(
                    f"读取的图像数据类型为 {img_low_res_uint.dtype}, 期望 uint16。将进行转换。",
                    UserWarning
                )
                img_low_res_uint = img_low_res_uint.astype(np.uint16)

            img_low_res_float = img_low_res_uint.astype(np.float64) / UINT16_MAX
            del img_low_res_uint # Free memory early

            check_input_image(img_low_res_float, 4, "estimate_global_A (float conversion)")

            typer.echo("  计算下采样图像的暗通道...")
            dark_low = dark_channel(img_low_res_float, dark_patch_sz) # Use potentially adjusted dark_patch_sz

            typer.echo("  估计大气光值...")
            global_A = atmospheric_light(img_low_res_float, dark_low)

            typer.echo(f"全局大气光 A 估算完成: {np.array2string(global_A, precision=4, floatmode='fixed')}")
            return global_A

    except FileNotFoundError:
        typer.echo(f"错误：输入文件未找到 {filepath}", err=True)
        return None
    except ValueError as ve:
        typer.echo(f"错误：处理 TIFF 文件时值错误: {ve}", err=True)
        return None
    except Exception as e:
        typer.echo(f"估算全局大气光 A 时发生未知错误: {e}", err=True)
        import traceback
        traceback.print_exc()
        return None


def process_tile(
    padded_tile_src_uint16: np.ndarray,
    global_A: np.ndarray,
    dark_patch_sz: int,
    transmission_patch_sz: int,
    guided_radius: int,
    guided_eps: float,
    omega: float,
    tx_threshold: float
    ) -> Optional[np.ndarray]:
    """
    对单个带重叠的块进行去雾处理 (uint16 输入, float64 处理, float64 输出 [0,1])

    Args:
        padded_tile_src_uint16: 输入的带重叠块 (HxWxC, uint16)。
        global_A: 估算的全局大气光 (1xN, float64)。
        dark_patch_sz: 暗通道patch大小。
        transmission_patch_sz: 初始透射率估计patch大小。
        guided_radius: 引导滤波半径。
        guided_eps: 引导滤波正则化系数。
        omega: 保留雾系数。
        tx_threshold: 透射率下限阈值。

    Returns:
        处理后的带重叠块 (HxWxC, float64, 范围 [0,1])，如果出错则返回 None。
    """
    try:
        # --- Input Validation and Conversion ---
        if padded_tile_src_uint16.dtype != np.uint16:
             padded_tile_src_uint16 = padded_tile_src_uint16.astype(np.uint16)

        padded_tile_I = padded_tile_src_uint16.astype(np.float64) / UINT16_MAX
        check_input_image(padded_tile_I, 4, "process_tile (input conversion)")
        del padded_tile_src_uint16 # Free memory

        # --- Dehazing Steps ---
        # 1. Initial Transmission Estimate
        #   Requires dark channel of normalized image
        te_t = transmission_estimate(padded_tile_I, global_A, transmission_patch_sz, omega)
        # Note: transmission_estimate internally calls dark_channel on normalized image

        # 2. Refine Transmission using Guided Filter
        #   Guide image is grayscale version of BGR channels from the original tile
        t_t = transmission_refine(padded_tile_I, te_t, guided_radius, guided_eps)

        # 3. Recover Scene Radiance
        J_tile_padded_float = recover(padded_tile_I, t_t, global_A, tx_threshold)

        return J_tile_padded_float

    except MemoryError:
        # Let the caller handle MemoryError specifically for better context
        raise MemoryError("Tile processing failed due to insufficient memory.")
    except ValueError as ve:
        # Log specific value errors encountered during processing
        typer.echo(f"\n错误: 处理块时发生值错误: {ve}", err=True)
        import traceback
        traceback.print_exc() # Print stack trace for ValueErrors as well
        return None # Allow loop to continue
    except Exception as e:
        # Catch any other unexpected errors during the core algorithm steps
        typer.echo(f"\n错误：处理块时发生未知错误: {e}", err=True)
        import traceback
        traceback.print_exc()
        return None # Allow loop to continue


def run_dehazing_pipeline(
    input_filepath: Path,
    output_dir: Path,
    tile_width: int,
    tile_height: int,
    overlap: int,
    save_tiles: bool,
    # Dehazing parameters
    dark_patch_sz: int,
    transmission_patch_sz: int,
    guided_radius: int,
    guided_eps: float,
    omega: float,
    tx_threshold: float,
    a_downsample_factor: int
):
    """
    执行完整的分块去雾流程

    Args:
        input_filepath: 输入 TIFF 图像文件路径。
        output_dir: 输出结果的目录。
        tile_width: 处理瓦片的宽度。
        tile_height: 处理瓦片的高度。
        overlap: 瓦片间的重叠像素数。
        save_tiles: True 则保存单个瓦片，False 则拼接为完整图像。
        dark_patch_sz: 暗通道patch大小。
        transmission_patch_sz: 初始透射率估计patch大小。
        guided_radius: 引导滤波半径。
        guided_eps: 引导滤波正则化系数。
        omega: 保留雾系数。
        tx_threshold: 透射率下限阈值。
        a_downsample_factor: 估算大气光时的下采样因子。
    """

    # --- 1. Estimate Global Atmospheric Light ---
    global_A = estimate_global_A(input_filepath,
                                 scale_factor=a_downsample_factor,
                                 dark_patch_sz=dark_patch_sz) # Pass dark patch size here too
    if global_A is None:
        typer.echo("无法估算全局大气光 A，处理中止。", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n开始分块去雾处理 (使用 Rasterio, {'保存瓦片' if save_tiles else '拼接完整图像'})...")

    # --- 2. Process Image Tile by Tile ---
    try:
        with rasterio.open(input_filepath) as src_dataset:
            H, W = src_dataset.height, src_dataset.width
            num_channels = src_dataset.count
            src_crs = src_dataset.crs
            src_transform = src_dataset.transform

            if num_channels != 4:
                 raise ValueError(f"错误：输入文件通道数 ({num_channels}) 不是 4。")
            if src_dataset.dtypes[0] != 'uint16':
                 warnings.warn(
                    f"输入文件数据类型为 {src_dataset.dtypes[0]}, 期望 uint16。处理仍将继续。",
                    UserWarning
                 )

            # --- Prepare Output Profile ---
            profile = src_dataset.profile # Get metadata from source
            profile.update(
                dtype=rasterio.uint16, # Output format
                count=num_channels,
                nodata=None, # Assume no nodata handling needed for dehazed result
                compress='lzw' # Add compression
            )

            # --- Setup Output (Single file or Tiles) ---
            dst_dataset = None
            output_filepath = None
            if not save_tiles:
                # Prepare single output file (tiled)
                output_filename = f"{input_filepath.stem}_dehazed.tif"
                output_filepath = output_dir / output_filename
                profile.update(tiled=True, blockxsize=tile_width, blockysize=tile_height)
                typer.echo(f"创建输出文件: {output_filepath}")
                dst_dataset = rasterio.open(output_filepath, 'w', **profile)
            else:
                # Ensure output dir exists for saving individual tiles
                 output_dir.mkdir(parents=True, exist_ok=True)
                 typer.echo(f"将独立保存瓦片到: {output_dir}")

            typer.echo(f"图像尺寸: {H}x{W}, 通道数: {num_channels}")
            typer.echo(f"分块设置: 瓦片={tile_width}x{tile_height}, 重叠={overlap}")

            # --- Calculate Tile Grid ---
            num_tiles_y = math.ceil(H / tile_height)
            num_tiles_x = math.ceil(W / tile_width)
            total_tiles = num_tiles_y * num_tiles_x
            typer.echo(f"总块数: {total_tiles} ({num_tiles_y} x {num_tiles_x})")

            # --- Iterate Through Tiles ---
            pbar = tqdm(total=total_tiles, desc="处理块", unit="块", file=sys.stdout)
            for y_idx in range(num_tiles_y):
                for x_idx in range(num_tiles_x):
                    tile_info_str = f"(块 {y_idx},{x_idx})"

                    # Calculate tile boundaries (write area)
                    y_start = y_idx * tile_height
                    y_end = min(y_start + tile_height, H)
                    x_start = x_idx * tile_width
                    x_end = min(x_start + tile_width, W)
                    write_width = x_end - x_start
                    write_height = y_end - y_start

                    # Calculate read area (with overlap)
                    read_y_start = max(0, y_start - overlap)
                    read_y_end = min(H, y_end + overlap)
                    read_x_start = max(0, x_start - overlap)
                    read_x_end = min(W, x_end + overlap)
                    read_width = read_x_end - read_x_start
                    read_height = read_y_end - read_y_start

                    read_window = rasterio.windows.Window(read_x_start, read_y_start, read_width, read_height)
                    write_window = rasterio.windows.Window(x_start, y_start, write_width, write_height)

                    # --- Read Tile ---
                    try:
                        padded_tile_chw = src_dataset.read(window=read_window)
                    except Exception as e:
                        pbar.write(f"\n错误：Rasterio 读取块 {tile_info_str} 时出错: {e}")
                        pbar.update(1)
                        continue

                    # Validate read data
                    if padded_tile_chw is None or padded_tile_chw.size == 0:
                        pbar.write(f"\n警告：Rasterio 读取块 {tile_info_str} 结果为空，跳过。")
                        pbar.update(1)
                        continue
                    if padded_tile_chw.shape != (num_channels, read_height, read_width):
                         pbar.write(f"\n警告：读取块 {tile_info_str} 形状不匹配 ({padded_tile_chw.shape})，预期 ({num_channels}, {read_height}, {read_width})，跳过。")
                         pbar.update(1)
                         continue

                    # Transpose to HWC for processing functions
                    padded_tile_hwc_uint16 = np.transpose(padded_tile_chw, (1, 2, 0))
                    del padded_tile_chw # Free memory

                    # --- Process Tile ---
                    J_tile_padded_float = None
                    try:
                        # Pass all dehazing parameters to process_tile
                        J_tile_padded_float = process_tile(
                            padded_tile_hwc_uint16, global_A,
                            dark_patch_sz, transmission_patch_sz,
                            guided_radius, guided_eps, omega, tx_threshold
                        )
                    except MemoryError:
                         pbar.write(f"\n错误：处理块 {tile_info_str} 时内存不足！请尝试减小瓦片尺寸或重叠。")
                         # Option: Stop processing completely
                         # raise MemoryError(f"Stopping due to memory error in tile {tile_info_str}.")
                    except Exception as e: # Catch other errors from process_tile
                         # Error message already printed inside process_tile
                         pass # Continue processing other tiles

                    del padded_tile_hwc_uint16 # Free memory

                    if J_tile_padded_float is None: # Skip if process_tile failed or returned None
                        pbar.update(1)
                        continue

                    # --- Extract Valid (Non-Overlapping) Area ---
                    inner_y_start = y_start - read_y_start
                    inner_y_end = inner_y_start + write_height
                    inner_x_start = x_start - read_x_start
                    inner_x_end = inner_x_start + write_width

                    # Basic check for slice validity
                    if not (0 <= inner_y_start < inner_y_end <= J_tile_padded_float.shape[0] and \
                            0 <= inner_x_start < inner_x_end <= J_tile_padded_float.shape[1]):
                         pbar.write(f"\n错误：计算块 {tile_info_str} 的有效区域索引时出错 ({inner_y_start}:{inner_y_end}, {inner_x_start}:{inner_x_end})。跳过写入。")
                         del J_tile_padded_float
                         pbar.update(1)
                         continue

                    J_tile_valid_float = J_tile_padded_float[inner_y_start:inner_y_end, inner_x_start:inner_x_end, :]
                    del J_tile_padded_float # Free memory

                    # Verify shape of extracted area
                    if J_tile_valid_float.shape != (write_height, write_width, num_channels):
                        pbar.write(f"\n警告：提取的块 {tile_info_str} 有效区域形状 ({J_tile_valid_float.shape}) 与预期 ({write_height, write_width, num_channels}) 不符，跳过写入。")
                        del J_tile_valid_float
                        pbar.update(1)
                        continue

                    # --- Convert to Output Format (uint16) ---
                    J_tile_out_hwc_uint16 = (J_tile_valid_float * UINT16_MAX).clip(0, UINT16_MAX).astype(np.uint16)
                    del J_tile_valid_float # Free memory

                    # Transpose back to CHW for Rasterio write
                    J_tile_out_chw = np.transpose(J_tile_out_hwc_uint16, (2, 0, 1))
                    del J_tile_out_hwc_uint16 # Free memory

                    # --- Write Tile ---
                    try:
                        if save_tiles:
                            # Save as individual tile file
                            tile_output_filename = f"{input_filepath.stem}_dehazed_tile_x{y_idx}_y{x_idx}.tif"
                            tile_output_path = output_dir / tile_output_filename

                            # Create profile specific to this tile
                            tile_profile = profile.copy() # Start with base profile (includes compression etc.)
                            tile_transform = rasterio.windows.transform(write_window, src_transform) # Get transform for this tile's origin
                            tile_profile.update(
                                height=write_height,
                                width=write_width,
                                transform=tile_transform,
                                tiled=False # Individual tiles usually don't need internal tiling
                            )
                            with rasterio.open(tile_output_path, 'w', **tile_profile) as tile_dst:
                                tile_dst.write(J_tile_out_chw)
                        else:
                            # Write to the main combined dataset
                            if dst_dataset:
                                dst_dataset.write(J_tile_out_chw, window=write_window)

                    except Exception as e:
                         pbar.write(f"\n错误：Rasterio 写入块 {tile_info_str} 时出错: {e}")
                    finally:
                        # Ensure memory is freed even if write fails
                        # (though J_tile_out_chw might be needed for retry logic if implemented)
                         if 'J_tile_out_chw' in locals(): del J_tile_out_chw

                    pbar.update(1) # Update progress bar for this tile

            pbar.close()
            # --- Finalize Output ---
            if dst_dataset:
                dst_dataset.close()
                typer.echo(f"\n所有块处理完成，输出文件已保存: {output_filepath}")
            elif save_tiles:
                 typer.echo(f"\n所有块处理完成，独立瓦片已保存到: {output_dir}")

    # --- Error Handling for Pipeline ---
    except rasterio.RasterioIOError as e:
        typer.echo(f"\nRasterio 文件 IO 错误: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
    except MemoryError as me:
         typer.echo(f"\n处理因内存不足而中止: {me}", err=True)
         raise typer.Exit(code=1)
    except ValueError as ve:
        typer.echo(f"\n处理因值错误而中止: {ve}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"\n处理过程中发生未捕获的严重错误: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
    finally:
        # Ensure dataset is closed even if error occurs mid-loop
        if dst_dataset and not dst_dataset.closed:
             dst_dataset.close()


# --- Typer App Setup ---
app = typer.Typer(
    help="使用暗通道先验对大型4通道TIFF影像进行分块去雾处理。",
    context_settings={"help_option_names": ["-h", "--help"]} # Add -h alias for help
)

@app.command()
def run_dehazing(
    # --- Input/Output Arguments ---
    input_path: Path = typer.Option(..., '-i', '--input-path',help="输入的待去雾栅格影像路径 (TIFF)。必需。",exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    output_dir: Path = typer.Option(..., '-o', '--output-dir',help="保存去雾后结果的文件夹路径。必需。",file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    tile_width: int = typer.Option(1024, "-tw", "--tile-width", min=64, help="处理瓦片的宽度（像素）。"),
    tile_height: int = typer.Option(1024, "-th", "--tile-height", min=64, help="处理瓦片的高度（像素）。"),
    overlap: int = typer.Option(128, "-ov", "--overlap", min=0,help="瓦片之间的重叠像素数。应小于瓦片尺寸。"),
    save_tiles: bool = typer.Option(False, "--save-tiles",help="是否仅保存每个处理后的瓦片，而不拼接保存完整图像。"),
    dark_patch_sz: int = typer.Option(15, "-dpsz", "--dark-patch-size", min=1,help="暗通道计算和全局A估计时使用的patch边长(奇数)。增大: 更平滑,不易受暗物体干扰,可能丢失细节/产生光晕。减小: 更多细节,对噪声/暗物体敏感。"),
    transmission_patch_sz: int = typer.Option(15, "-tpsz", "--transmission-patch-size", min=1,help="初始透射率估计时使用的patch边长(奇数)。增大: 初始透射率更平滑。减小: 初始透射率细节更多,可能块效应。"),
    guided_radius: int = typer.Option(60, "-gr", "--guided-radius", min=1,help="引导滤波窗口半径。增大: 透射率图更平滑,减少光晕,可能模糊边缘。减小: 保留更多细节和噪声,可能留有光晕。"),
    guided_eps: float = typer.Option(0.0001, "-geps", "--guided-eps", min=1e-9,help="引导滤波正则化系数(epsilon)。增大: 滤波效果更平滑(模糊)。减小: 更贴近原始结构(锐利)。"),
    omega: float = typer.Option(0.95, "-om", "--omega", min=0.1, max=1.0,help="去雾强度因子(1-omega*dark_channel)。增大(接近1): 去雾更彻底。减小: 保留更多雾感,可能更自然。"),
    tx_threshold: float = typer.Option(0.1, "-tx", "--transmission-threshold", min=0.01, max=0.9,help="透射率下限阈值 t0。增大: 限制最大去雾程度,避免过度增强噪声/伪影。减小: 允许在浓雾区去雾更彻底,可能引入噪声。"),
    a_downsample_factor: int = typer.Option(8, "-adsf", "--a-downsample-factor", min=1,help="估算全局大气光A时的下采样因子。增大: 更快,内存占用更少,可能损失精度。减小(接近1): 更慢,内存占用多,可能更精确但易受亮物体影响。"),
):
    typer.echo(f"输入影像: {input_path}")
    typer.echo(f"输出目录: {output_dir}")
    typer.echo(f"瓦片尺寸: {tile_width}x{tile_height}")
    typer.echo(f"重叠: {overlap}")
    typer.echo(f"仅保存瓦片: {'是' if save_tiles else '否'}")
    typer.echo(f"暗通道 Patch Size: {dark_patch_sz}")
    typer.echo(f"透射率 Patch Size: {transmission_patch_sz}")
    typer.echo(f"引导滤波半径: {guided_radius}")
    typer.echo(f"引导滤波 Epsilon: {guided_eps}")
    typer.echo(f"Omega (去雾强度): {omega}")
    typer.echo(f"透射率阈值 (t0): {tx_threshold}")
    typer.echo(f"大气光下采样因子: {a_downsample_factor}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
         typer.echo(f"错误：无法创建输出目录 {output_dir}: {e}", err=True)
         raise typer.Exit(code=1)

    if overlap >= tile_width or overlap >= tile_height:
        typer.echo(f"错误：重叠 ({overlap}) 不能大于或等于瓦片尺寸 ({tile_width}x{tile_height})。", err=True)
        raise typer.Exit(code=1)

    run_dehazing_pipeline(
        input_filepath=input_path,
        output_dir=output_dir,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        save_tiles=save_tiles,
        dark_patch_sz=dark_patch_sz,
        transmission_patch_sz=transmission_patch_sz,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
        omega=omega,
        tx_threshold=tx_threshold,
        a_downsample_factor=a_downsample_factor
    )

    typer.echo("\n脚本执行完毕。")

if __name__ == '__main__':
    app()