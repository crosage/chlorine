import numpy as np
import rasterio

# 输入文件路径
input_file = "D:/building/test/GF1C_PMS_E118.6_N37.4_20240603_L1A1022262224-PAN_ortho_fuse.tif"
output_file = "D:/building/test/GF1C_PMS_E118.6_N37.4_20240603_L1A1022262224-PAN_ortho_fuse_uint16.tif"

# 打开输入文件
with rasterio.open(input_file) as src:
    # 创建输出文件配置
    profile = src.profile

    # 确保nodata值在uint16范围内，或移除它
    if 'nodata' in profile and (profile['nodata'] is None or profile['nodata'] > 65535):
        profile.pop('nodata')  # 移除 nodata 属性
    profile.update(dtype=rasterio.uint16, count=src.count)

    # 打开输出文件
    with rasterio.open(output_file, 'w', **profile) as dst:
        for i in range(1, src.count + 1):
            # 读取 uint32 数据
            data = src.read(i)

            # 确保数据是 uint32 类型
            if data.dtype != np.uint32:
                raise ValueError("输入数据类型不是 uint32！")

            # 将 uint32 数据映射到 uint16
            data_uint16 = ((data / 4294967295.0) * 65535).astype(np.uint16)

            # 将映射后的数据写入新文件
            dst.write(data_uint16, i)

print(f"文件已成功映射并保存为 {output_file}")
