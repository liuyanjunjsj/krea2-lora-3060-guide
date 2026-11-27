# -*- coding: utf-8 -*-
"""
apply_patches.py — 给 ai-toolkit v0.13.43 自动打 4 个补丁（幂等，可重复执行）

用法:
    python apply_patches.py --ai-toolkit-dir <ai-toolkit 目录>

补丁清单见 docs/补丁详解.md。打补丁前请确认你的 ai-toolkit 版本
（pip show ai-toolkit 或 git log）；其他版本字符串可能对不上，
届时按补丁详解手动改。

仅修改源码文件，不联网、不装依赖。
"""

import argparse
import shutil
import sys
from pathlib import Path

P1_IMPORT_ANCHOR = "QWEN_IMAGE_VAE_PATH = "
P1_IMPORT_NEW = (
    "QWEN_IMAGE_VAE_PATH = "
    "\n# [patch1] 借用官方 _markers_from_header 在读盘后补合成标记"
    "\nfrom toolkit.models.v2._mixin import _markers_from_header"
)

P1_REPLACEMENTS = [
    # (旧, 新) —— krea2 加载器所有 load_file 返回处包 _markers_from_header
    (
        "        return load_file(name_or_path)\n",
        "        return _markers_from_header(name_or_path, load_file(name_or_path))\n",
    ),
    (
        "            return load_file(os.path.join(name_or_path, filename))\n",
        "            path = os.path.join(name_or_path, filename)\n"
        "            return _markers_from_header(path, load_file(path))\n",
    ),
    (
        "            return load_file(os.path.join(name_or_path, candidates[0]))\n",
        "            path = os.path.join(name_or_path, candidates[0])\n"
        "            return _markers_from_header(path, load_file(path))\n",
    ),
    (
        "\n    return load_file(path)\n",
        "\n    return _markers_from_header(path, load_file(path))\n",
        True,   # 该短锚点可能在文件其他函数出现，仅唯一时才自动改
    ),
]

P2_OLD = (
    '        if self.layer_offloading and self.qtype_te == "qfloat8":\n'
    '            self.qtype_te = "float8"\n'
)
P2_NEW = (
    "        # [patch2] 保留 TE 的 qfloat8(quanto)，不再强转 float8(torchao)：\n"
    "        # torchao 在低显存训练进程内逐块量化必炸（显存空闲 10.8G 也 OOM），\n"
    "        # quanto 已实测 36 块全过\n"
    '        # if self.layer_offloading and self.qtype_te == "qfloat8":\n'
    '        #     self.qtype_te = "float8"\n'
)

P3_ANCHOR = (
    "            q_device = quantize_device if quantize_device is not None else device\n"
)
P3_NEW = (
    "            q_device = quantize_device if quantize_device is not None else device\n"
    "            # [patch3] low_vram（组件驻留 CPU）时强制纯 CPU 量化：\n"
    "            # 训练进程内做 GPU 逐块量化会在显存空闲 10.8G 时报 CUDA OOM\n"
    "            # （独立进程同样调用正常）。quanto 纯 CPU 量化已验证（36 块全过）。\n"
    '            if device is not None and torch.device(device).type == "cpu" and offload:\n'
    "                q_device = None\n"
    "                keep_on_device = False\n"
)

P4_ANCHOR = (
    "if TYPE_CHECKING:\n"
    "    from .manager import MemoryManager\n"
)
P4_NEW = (
    "if TYPE_CHECKING:\n"
    "    from .manager import MemoryManager\n"
    "\n"
    "# [patch4] Windows 上对大模型全量锁页（pin_memory ≈ 12G 底座 + 4.5G 编码器）\n"
    "# 会耗尽内核非分页池：之后进程内任何新的 CUDA 分配——哪怕几十 MB、显存空闲\n"
    "# 10G+——都报 CUDA OOM。设 AI_TOOLKIT_NO_PIN=1 跳过全部锁页。\n"
    '_NO_PIN = os.environ.get("AI_TOOLKIT_NO_PIN", "0") == "1"\n'
)

P4_REPLACEMENTS = [
    (
        "            and not inner.is_pinned()\n        ):\n",
        "            and not inner.is_pinned()\n            and not _NO_PIN\n        ):\n",
    ),
    (
        "    if torch.cuda.is_available():\n        try:\n            t = t.pin_memory()\n",
        "    if torch.cuda.is_available() and not _NO_PIN:\n        try:\n            t = t.pin_memory()\n",
    ),
    (
        "if torch.cuda.is_available() and not buf.is_pinned():\n",
        "if torch.cuda.is_available() and not buf.is_pinned() and not _NO_PIN:\n",
    ),
]


def apply(text, old, new, label, applied_flags, require_unique=False):
    if require_unique and text.count(old) != 1:
        print(f"  [!] {label}: 锚点出现 {text.count(old)} 次，不敢自动改，请按补丁详解手动改这一处")
        return text
    if old not in text:
        print(f"  [=] {label}: 未找到旧代码（大概率已打过），跳过")
        return text
    applied_flags.append(label)
    print(f"  [+] {label}: 已打")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-toolkit-dir", required=True)
    args = ap.parse_args()
    root = Path(args.ai_toolkit_dir)
    if not root.is_dir():
        sys.exit(f"目录不存在: {root}")

    changed = []

    # ---- 补丁 1 ----
    p1 = root / "extensions_built_in/diffusion_models/krea2/krea2.py"
    print(f"补丁 1 (krea2 fp8 识别): {p1}")
    text = p1.read_text(encoding="utf-8")
    if "_markers_from_header" in text:
        print("  [=] P1: 已打过（检测到 _markers_from_header），跳过")
    else:
        text = apply(text, P1_IMPORT_ANCHOR, P1_IMPORT_NEW, "P1-import", changed)
        for i, item in enumerate(P1_REPLACEMENTS, 1):
            old, new = item[0], item[1]
            unique = item[2] if len(item) > 2 else False
            text = apply(text, old, new, f"P1-load#{i}", changed, require_unique=unique)
        p1.write_text(text, encoding="utf-8")

    # ---- 补丁 2 ----
    p2 = root / "toolkit/config_modules.py"
    print(f"补丁 2 (TE 保留 quanto): {p2}")
    text = p2.read_text(encoding="utf-8")
    orig = text
    text = apply(text, P2_OLD, P2_NEW, "P2-te-quanto", changed)
    if text != orig:
        p2.write_text(text, encoding="utf-8")

    # ---- 补丁 3 ----
    p3 = root / "toolkit/models/v2/_mixin.py"
    print(f"补丁 3 (编码器纯 CPU 量化): {p3}")
    text = p3.read_text(encoding="utf-8")
    if "keep_on_device = False" in text and "q_device = None" in text:
        print("  [=] P3: 已打过（检测到纯 CPU 量化分支），跳过")
    else:
        text = apply(text, P3_ANCHOR, P3_NEW, "P3-cpu-quant", changed)
        p3.write_text(text, encoding="utf-8")

    # ---- 补丁 4 ----
    p4 = root / "toolkit/memory_management/manager_modules.py"
    print(f"补丁 4 (NO_PIN 开关): {p4}")
    text = p4.read_text(encoding="utf-8")
    if "_NO_PIN = os.environ.get" in text:
        print("  [=] P4: 已打过（检测到 _NO_PIN 常量），跳过")
    else:
        text = apply(text, P4_ANCHOR, P4_NEW, "P4-const", changed)
        for i, (old, new) in enumerate(P4_REPLACEMENTS, 1):
            text = apply(text, old, new, f"P4-gate#{i}", changed)
        p4.write_text(text, encoding="utf-8")

    print("\n完成。本次实际修改: " + (", ".join(changed) if changed else "无（全部已打过）"))
    print("建议先备份再跑: python -c \"import shutil; shutil.copytree(r'<ai-toolkit>', r'<ai-toolkit>.bak')\"")
    print("（本脚本自带幂等保护，但防手滑永远不亏）")


if __name__ == "__main__":
    main()
