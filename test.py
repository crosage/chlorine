import numpy as np
import tifffile

def get_tiff_rgb_range_tifffile(tiff_filepath):
    try:
        # 使用 tifffile 读取图像，直接得到 NumPy 数组
        img_array = tifffile.imread(tiff_filepath)
        dtype = img_array.dtype

        # 后续处理与 Pillow 方法类似...
        if img_array.ndim == 3 and img_array.shape[-1] >= 3: # 假设通道在最后维度
            # 根据实际的通道顺序调整索引 [:,:,0] 或 [0,:,:] 等
            # 例如，如果 shape 是 (channels, height, width)
            if img_array.shape[0] == 3 or img_array.shape[0] == 4:
                 r_channel = img_array[0, :, :]
                 g_channel = img_array[1, :, :]
                 b_channel = img_array[2, :, :]
            # 例如，如果 shape 是 (height, width, channels)
            elif img_array.shape[2] == 3 or img_array.shape[2] == 4:
                 r_channel = img_array[:, :, 0]
                 g_channel = img_array[:, :, 1]
                 b_channel = img_array[:, :, 2]
            else:
                return f"错误：无法确定 RGB 通道维度 {img_array.shape}"

            ranges = {
                'R': (np.min(r_channel), np.max(r_channel)),
                'G': (np.min(g_channel), np.max(g_channel)),
                'B': (np.min(b_channel), np.max(b_channel)),
                'dtype': str(dtype)
            }
            return ranges
        elif img_array.ndim == 2:
            return {
                'Grayscale': (np.min(img_array), np.max(img_array)),
                'dtype': str(dtype)
            }
        else:
             return f"错误：不支持的图像维度 {img_array.shape}"
    except FileNotFoundError:
        return f"错误：文件未找到 '{tiff_filepath}'"
    except Exception as e:
        return f"处理文件 '{tiff_filepath}' 时出错: {e}"

# --- 使用示例 ---
file_path = "D:\Train\origin\GF1_PMS1_E51.6_N49.6_20231106_L1A13154817001-MSS1_fuse.tiff"
result = get_tiff_rgb_range_tifffile(file_path)
# ... (后续打印部分与 Pillow 示例相同) ...
print(result)