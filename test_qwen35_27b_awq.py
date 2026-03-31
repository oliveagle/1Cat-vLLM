#!/usr/bin/env python3
"""
简单测试脚本：测试 tclf90/Qwen3.5-27B-AWQ 模型
"""

import argparse
import sys
import time
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="测试 Qwen3.5-27B-AWQ 模型"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-27B-AWQ",
        help="模型路径"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="最大模型长度"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU 内存利用率"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="用一句话回答：2+2等于几？",
        help="测试提示词"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="最大生成 token 数"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("1Cat-vLLM - Qwen3.5-27B-AWQ 测试")
    print("=" * 80)
    print()
    print(f"模型路径: {args.model_path}")
    print(f"张量并行: {args.tensor_parallel_size}")
    print(f"最大模型长度: {args.max_model_len}")
    print(f"GPU 内存利用率: {args.gpu_memory_utilization}")
    print()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"错误: 无法导入 vllm: {e}")
        print()
        print("请确保:")
        print("1. 已激活正确的虚拟环境")
        print("2. 已安装 1Cat-vLLM")
        print("3. 已安装 PyTorch 和 CUDA")
        sys.exit(1)

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    print("正在加载模型...")
    print("(注意: 第一次加载可能需要 1-3 分钟来编译内核)")
    print()

    start_load = time.time()
    try:
        llm = LLM(
            model=args.model_path,
            quantization="awq",
            dtype="float16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            # 1Cat-vLLM 推荐配置
            skip_mm_profiling=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            attention_backend="TRITON_ATTN",
            compilation_config={
                "cudagraph_mode": "full_and_piecewise",
                "cudagraph_capture_sizes": [1],
            },
        )
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    load_time = time.time() - start_load

    print()
    print("=" * 80)
    print(f"模型加载成功！(耗时: {load_time:.2f} 秒)")
    print("=" * 80)
    print()

    # 测试提示词
    prompts = [
        args.prompt,
        "请介绍一下你自己。",
        "写一个简单的 Python 函数来计算斐波那契数列。",
    ]

    for i, prompt in enumerate(prompts, 1):
        print()
        print("-" * 80)
        print(f"测试 {i}/{len(prompts)}")
        print("-" * 80)
        print(f"提示词: {prompt}")
        print()
        print("生成中...")
        print()

        try:
            start_gen = time.time()
            outputs = llm.generate(prompt, sampling_params)
            gen_time = time.time() - start_gen

            for output in outputs:
                prompt_text = output.prompt
                generated_text = output.outputs[0].text
                num_tokens = len(output.outputs[0].token_ids)
                print(f"输入: {prompt_text}")
                print(f"输出: {generated_text}")
                print(f"生成 token 数: {num_tokens}")
                print(f"生成耗时: {gen_time:.2f} 秒")
                if num_tokens > 0 and gen_time > 0:
                    print(f"生成速度: {num_tokens/gen_time:.2f} tok/s")
                print()

        except Exception as e:
            print(f"错误: 生成失败: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
