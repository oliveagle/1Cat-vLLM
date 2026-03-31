#!/usr/bin/env python3
"""
快速验证脚本：验证 vLLM + AWQ 环境是否正常工作
"""

import sys
import time

print("=" * 80)
print("1Cat-vLLM 环境快速验证")
print("=" * 80)
print()

# 1. 检查导入
print("[1/5] 检查导入...")
try:
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    print("  ✓ PyTorch 版本:", torch.__version__)
    print("  ✓ vLLM 版本:", vllm.__version__)
    print("  ✓ 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 2. 检查 CUDA
print("[2/5] 检查 CUDA...")
try:
    print("  ✓ CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  ✓ CUDA 版本:", torch.version.cuda)
        print("  ✓ GPU 数量:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  ✓ GPU {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print(f"  ✗ CUDA 检查失败: {e}")

print()

# 3. 检查模型文件
print("[3/5] 检查模型文件...")
model_path = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-27B-AWQ"
import os
if os.path.exists(model_path):
    print(f"  ✓ 模型路径存在: {model_path}")
    files = os.listdir(model_path)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    print(f"  ✓ 找到 {len(safetensors)} 个 safetensors 文件")
    if "config.json" in files:
        print("  ✓ 找到 config.json")
    if "tokenizer.json" in files:
        print("  ✓ 找到 tokenizer.json")
else:
    print(f"  ✗ 模型路径不存在: {model_path}")
    sys.exit(1)

print()

# 4. 加载模型配置进行验证
print("[4/5] 验证模型配置...")
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    print(f"  ✓ 模型类型: {config.model_type}")
    print(f"  ✓ 隐藏层大小: {config.hidden_size}")
    print(f"  ✓ 层数: {config.num_hidden_layers}")
    print(f"  ✓ 注意力头数: {config.num_attention_heads}")
    print(f"  ✓ 词汇表大小: {config.vocab_size}")
except Exception as e:
    print(f"  ✗ 配置加载失败: {e}")

print()
print("=" * 80)
print("环境验证完成！")
print("=" * 80)
print()
print("提示: 现在可以运行完整测试了:")
print("  python test_qwen35_27b_awq.py")
print()
print("或者运行 API 服务:")
print("  python -m vllm.entrypoints.openai.api_server \\")
print(f"    --model {model_path} \\")
print("    --quantization awq \\")
print("    --dtype float16 \\")
print("    --tensor-parallel-size 1 \\")
print("    --max-model-len 4096 \\")
print("    --skip-mm-profiling \\")
print("    --attention-backend TRITON_ATTN")
print()
