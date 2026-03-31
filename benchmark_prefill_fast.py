#!/usr/bin/env python3
"""
快速预填充压测脚本 - 快速测试几个关键点
"""

import argparse
import random
import time
import sys

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-27B-AWQ 快速预填充压测"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-27B-AWQ",
        help="模型路径"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("1Cat-vLLM - Qwen3.5-27B-AWQ 快速预填充压测")
    print("=" * 80)
    print()

    # 导入 vLLM
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
    except ImportError as e:
        print(f"错误: 无法导入 vllm: {e}")
        sys.exit(1)

    # 加载 tokenizer
    print("正在加载 tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"  ✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        sys.exit(1)

    print()

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
    )

    print("正在加载模型...")
    print()

    try:
        llm = LLM(
            model=args.model_path,
            quantization="awq",
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
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

    print()
    print("模型加载完成!")
    print()

    # 测试长度
    test_lengths = [256, 512, 1024, 2048, 4000]
    vocab_size = tokenizer.vocab_size

    print("=" * 80)
    print("开始压测...")
    print("=" * 80)
    print()

    results = []

    for prompt_len in test_lengths:
        # 生成随机 prompt
        prompt_tokens = [random.randint(100, vocab_size - 100) for _ in range(prompt_len)]
        prompts = [TokensPrompt(prompt_token_ids=prompt_tokens)]

        # 热身 2 次
        print(f"长度 {prompt_len}: 热身...")
        for _ in range(2):
            llm.generate(prompts, sampling_params)

        # 测试 3 次
        times = []
        for i in range(3):
            start = time.perf_counter()
            llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            throughput = prompt_len / elapsed
            print(f"  测试 {i+1}: {elapsed:.3f}s, {throughput:.2f} tok/s")

        avg_time = sum(times) / len(times)
        avg_throughput = prompt_len / avg_time
        max_throughput = prompt_len / min(times)

        print(f"  → 平均: {avg_time:.3f}s, {avg_throughput:.2f} tok/s (最好: {max_throughput:.2f} tok/s)")
        print()

        results.append({
            "length": prompt_len,
            "avg_throughput": avg_throughput,
            "max_throughput": max_throughput,
        })

    # 汇总结果
    print("=" * 80)
    print("预填充压测结果汇总")
    print("=" * 80)
    print()
    print(f"{'输入长度':<12} {'平均(tok/s)':<18} {'最好(tok/s)':<18}")
    print("-" * 48)
    for r in results:
        print(f"{r['length']:<12} {r['avg_throughput']:<18.2f} {r['max_throughput']:<18.2f}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
