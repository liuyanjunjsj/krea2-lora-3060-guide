# Krea2 风格 LoRA 训练 · RTX 3060 12G + 32G 内存 生存指南

在 **RTX 3060 12G / 32G 内存 / Windows 11** 上用 [ai-toolkit](https://github.com/ostris/ai-toolkit)（v0.13.43）训练 Krea 2 风格 LoRA 的完整方案。

官方文档不会告诉你的四件事，这里全部踩过、全部修好：

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | ComfyUI fp8 预量化底座识别 | 26G bf16 底座在 32G 内存机器上加载即卡死 | 换 fp8 预量化底座 + 补一个标记合成调用（详见 [补丁 1](docs/补丁详解.md)） |
| 2 | 开内存管家后编码器量化库被强转 | 显存空闲 10.8G 却报 CUDA OOM | TE 保留 quanto、强制纯 CPU 量化（补丁 2、3） |
| 3 | 内存管家锁页打爆内核非分页池 | 显存空闲 10G+，进程内**任何** CUDA 分配全失败 | 环境变量 `AI_TOOLKIT_NO_PIN=1`（补丁 4） |
| 4 | 权重被 Windows 分页到硬盘 | 磁盘 100%、显卡时动时不动、步速持续恶化 | 部分层常驻显卡 + 重开文本嵌入缓存 |

**实测性能**（3060 12G，fp8 底座，rank 32，2500 步）：
- 优化后 **约 24 秒/步**，显卡利用率稳定 100%，2500 步约 **17 小时**
- 不打补丁不调参的话：要么跑不起来，要么磁盘 100% 卡成 60+ 秒/步（2 天半）

> 完整排障故事见 [踩坑实录](docs/踩坑实录.md)，每处源码改动见 [补丁详解](docs/补丁详解.md)。

## 硬件 / 软件前提

- GPU：RTX 3060 12G（显存贴满跑，余量约 280MB）
- 内存：**32G 是底线**（模型权重 + 系统约 28G）；64G 更从容
- 系统：Windows 11（Linux 未验证，理论上同样适用）
- Python 3.13 + CUDA 13.0（ai-toolkit 官方 venv）
- 磁盘：模型约 30G + 训练输出

## 快速开始

```bat
:: 1. 克隆本仓库，进入目录
:: 2. 安装 ai-toolkit（官方流程）到 lora-training\ai-toolkit
:: 3. 打四个补丁（幂等，重复执行无副作用）
python scripts\apply_patches.py --ai-toolkit-dir lora-training\ai-toolkit

:: 4. 下载模型（fp8 底座 12.2G + Qwen3-VL-4B 编码器 + Qwen-Image VAE）
powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1

:: 5. 准备训练集（见下文），改 config 里的路径
:: 6. 关掉所有占显存/内存的程序，双击
scripts\一键训练_Krea2风格LoRA.bat
```

## 模型清单（全部国内直连，无需梯子）

| 模型 | 用途 | 大小 | 来源 |
|---|---|---|---|
| `krea2_raw_fp8_scaled.safetensors` | 训练底座（预量化 fp8） | 12.2G | [ModelScope Comfy-Org/Krea-2](https://modelscope.cn/models/Comfy-Org/Krea-2) |
| Qwen3-VL-4B-Instruct | 文本编码器（bf16 完整目录） | ~9G | [ModelScope Qwen/Qwen3-VL-4B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen-Image VAE | VAE（diffusers 命名目录） | ~0.5G | [ModelScope Qwen/Qwen-Image](https://modelscope.cn/models/Qwen/Qwen-Image) 的 `vae/` 子目录 |

⚠️ **不要下载 bf16 原版底座**（`raw.safetensors` 26.3G）：32G 内存机器加载峰值 29G+，必卡死。fp8 版键名与原版完全一致，训练器原生兼容（补丁 1 之后）。

⚠️ VAE 必须是 **diffusers 命名**（`config.json` + `diffusion_pytorch_model.safetensors`），不是 ComfyUI 命名。

## 训练集

```
dataset/clean/
├── 001.jpg
├── 001.txt      ← caption，每张图一个同名 txt
├── 002.jpg
├── 002.txt
└── ...
```

- 规模：我们用了 **57 张**同风格图，足够
- caption 统一以触发词开头（我们用 `krea2style`），后面写画面内容
- 分辨率桶：`[512, 768, 1024]`
- ⚠️ 本仓库**不含训练图**（版权原因），格式照上面准备即可

## 关键配置（`config/krea2_style_lora_3060.yml`）

- `qtype: "float8_e4m3fn"`：必须与 fp8 底座文件的实际量化后端一致，写 `qfloat8` 会触发反量化→内存爆
- `layer_offloading: true` + `layer_offloading_transformer_percent: 0.8`：20% 层常驻显卡
- `cache_text_embeddings: true`：缓存完卸载编码器，省 4.5G 内存（依赖补丁 4）
- `low_vram: true`、`quantize_te: true`、`qtype_te: "qfloat8"`
- 启动时**必须**带环境变量 `AI_TOOLKIT_NO_PIN=1`（bat 脚本已内置）

## 训练节奏参考

| 阶段 | 耗时 |
|---|---|
| 底座加载 + fp8 识别 | ~1 分钟 |
| 编码器纯 CPU 量化 | ~32 秒 |
| VAE 缓存（57 图） | ~2 分钟 |
| 文本嵌入缓存（一次性） | ~3 分钟 |
| 正式训练 | ~24 秒/步 × 2500 步 ≈ 17 小时 |

每 250 步自动存档（`output/krea2_style_lora/`），中断可续。

## FAQ

**Q：显存贴满 12G 会不会炸？**
贴满是设计好的（20% 层常驻 + 激活 + LoRA）。若报 OOM，把 `layer_offloading_transformer_percent` 从 0.8 调回 0.9，牺牲一点速度。

**Q：为什么不用 Comfy-Org 的 int8 版（`krea2_raw_int8_convrot`）？**
fp8 是浮点量化，与 ai-toolkit 的 float8 后端原生对齐；int8 需要额外转换路径，未验证。

**Q：256G 显存的卡（4090 等）需要这套吗？**
不需要。`layer_offloading` 关掉、bf16 底座直接上，那是另一条更简单的路。本项目专治 12G 卡 + 32G 内存。

**Q：训练出来怎么用？**
输出是 ai-toolkit 格式 safetensors，ComfyUI 直接放 `models/loras/` 使用，触发词写在 prompt 里。

## 致谢与出处

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) —— 训练器本体
- [Comfy-Org/Krea-2](https://modelscope.cn/models/Comfy-Org/Krea-2) —— fp8 预量化底座重打包
- 黑魔法研究：飞书多 AI 协作排查（WorkBuddy 主控 + 多轮隔离实验），2026-11-27 实战记录

## License

MIT（代码与文档）。训练图、生成立绘等素材不在本仓库范围。
