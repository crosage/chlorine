import os
import subprocess
from pathlib import Path

# === 需要修改：你的去雾脚本路径 ===
DEHAZE_SCRIPT = r"D:\code\shpdealer\dehazer.py"  # ←改成你的实际路径

# === 需要修改：数据根目录 ===
ROOT = Path(r"X:\home\F\zzn\Kazakhstan\data")

# 固定参数（质量优先）
PARAMS = [
    "-tw", "768",
    "-th", "768",
    "-ov", "160",
    "-dpsz", "21",
    "-tpsz", "21",
    "-gr", "90",
    "-geps", "0.0002",
    "-om", "0.97",
    "-tx", "0.08",
    "-adsf", "4",
]

def is_target_raw(p: Path) -> bool:
    """是否为需要处理的原始影像"""
    name = p.name.lower()
    if not (name.endswith(".tif") or name.endswith(".tiff")):
        return False
    if name.startswith("result_"):
        return False
    if name.startswith("dehaze_"):
        return False
    return True

def main():
    all_files = list(ROOT.rglob("*.tif")) + list(ROOT.rglob("*.tiff"))
    all_files = [f for f in all_files if is_target_raw(f)]

    print(f"✅ Found {len(all_files)} images to process")

    for src in all_files:
        folder = src.parent
        # 目标最终文件名：dehaze_<原文件名>（保持原扩展名）
        final_out = folder / f"dehaze_{src.name}"
        if final_out.exists():
            print(f"⏭️ Skip exists: {final_out.name}")
            continue

        print(f"🚀 Processing: {src.name}")

        # 调用你的 dehazer.py（注意：无子命令，不能带 run-dehazing）
        cmd = [
            "python", DEHAZE_SCRIPT,
            "-i", str(src),
            "-o", str(folder),
        ] + PARAMS

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed (runtime error): {src.name} — {e}")
            continue
        except Exception as e:
            print(f"❌ Failed (unknown): {src.name} — {e}")
            continue

        # 你的脚本会在输出目录生成：<stem>_dehazed.tif
        tmp_out = folder / f"{src.stem}_dehazed.tif"
        if not tmp_out.exists():
            print(f"⚠️ Output not found (expected): {tmp_out.name}")
            # 有些人会把扩展名写成 .tiff，这里也尝试一下
            alt_tmp = folder / f"{src.stem}_dehazed.tiff"
            if alt_tmp.exists():
                tmp_out = alt_tmp
            else:
                print("❌ No dehazed output to rename, skip.")
                continue

        # 把临时输出改名为目标前缀 dehaze_<原文件名>，并保持原扩展名
        try:
            # 如果原始是 .tiff 而生成是 .tif，就改扩展名以匹配原始
            desired_ext = src.suffix  # 包含点
            final_out = final_out.with_suffix(desired_ext)
            tmp_out.rename(final_out)
            print(f"✅ Saved: {final_out.name}")
        except Exception as e:
            print(f"⚠️ Rename failed ({tmp_out.name} -> {final_out.name}): {e}")

    print("\n🎯 All done.")

if __name__ == "__main__":
    main()
