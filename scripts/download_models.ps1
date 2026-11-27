# download_models.ps1 — Krea2 LoRA 训练模型下载（国内直连，无需梯子）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1 -Root <模型根目录>
# 下载前确认磁盘剩余 >= 25G

param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = "Stop"

# 1) fp8 预量化底座（12.2G）— ModelScope 直连
$baseDir = Join-Path $Root "Krea-2-Raw"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null
$baseFile = Join-Path $baseDir "krea2_raw_fp8_scaled.safetensors"
if (Test-Path $baseFile) {
    Write-Host "[跳过] 底座已存在: $baseFile"
} else {
    Write-Host "[下载] fp8 底座 12.2G ..."
    curl.exe -L -C - --retry 5 --retry-delay 10 `
        -o $baseFile `
        "https://modelscope.cn/models/Comfy-Org/Krea-2/resolve/master/diffusion_models/krea2_raw_fp8_scaled.safetensors"
}

# 2) Qwen3-VL-4B-Instruct 编码器（完整 bf16 目录）— 用 modelscope CLI
#    首次使用: pip install modelscope
$teDir = Join-Path $Root "Qwen3-VL-4B-Instruct"
if (Test-Path (Join-Path $teDir "config.json")) {
    Write-Host "[跳过] 编码器已存在: $teDir"
} else {
    Write-Host "[下载] Qwen3-VL-4B-Instruct 完整目录（约 9G）..."
    modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir $teDir
}

# 3) Qwen-Image VAE（diffusers 命名目录）
$vaeDir = Join-Path $Root "Qwen-Image-Vae"
if (Test-Path (Join-Path $vaeDir "diffusion_pytorch_model.safetensors")) {
    Write-Host "[跳过] VAE 已存在: $vaeDir"
} else {
    Write-Host "[下载] Qwen-Image VAE ..."
    New-Item -ItemType Directory -Force -Path $vaeDir | Out-Null
    modelscope download --model Qwen/Qwen-Image --include "vae/*" --local_dir (Join-Path $Root "_qwen_image_tmp")
    Move-Item (Join-Path $Root "_qwen_image_tmp\vae\*") $vaeDir
    Remove-Item -Recurse -Force (Join-Path $Root "_qwen_image_tmp")
}

Write-Host ""
Write-Host "全部完成。模型目录结构应为:"
Write-Host "  $Root\Krea-2-Raw\krea2_raw_fp8_scaled.safetensors   (约 12.2G)"
Write-Host "  $Root\Qwen3-VL-4B-Instruct\  (完整目录，含 config.json)"
Write-Host "  $Root\Qwen-Image-Vae\        (config.json + diffusion_pytorch_model.safetensors)"
Write-Host "把这三个路径填进 config/krea2_style_lora_3060.yml 对应字段即可。"
