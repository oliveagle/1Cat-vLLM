#!/usr/bin/env python3
"""
预填充 (Prefill) 性能压测脚本
测试不同输入长度下的预填充吞吐量
"""

import argparse
import random
import time
import sys
from typing import List, Tuple

import torch
from tqdm import tqdm


def generate_random_prompt(tokenizer, length: int) -> List[int]:
    """生成指定长度的随机 prompt"""
    # 使用真实词汇范围内的 token ID
    vocab_size = tokenizer.vocab_size
    # 避免特殊 token
    return [random.randint(100, vocab_size - 100) for _ in range(length)]


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-27B-AWQ 预填充性能压测"
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
        default=4096,
        help="最大模型长度"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU 内存利用率"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="热身次数"
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="每个长度测试次数"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("1Cat-vLLM - Qwen3.5-27B-AWQ 预填充性能压测")
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
        print(f"  ✓ Tokenizer 加载成功, 词汇表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        sys.exit(1)

    print()

    # 测试的输入长度
    test_lengths = [64, 128, 256, 512, 1024, 2048, 3072, 4000]
    print(f"测试输入长度: {test_lengths}")
    print(f"热身次数: {args.num_warmup}")
    print(f"每个长度测试次数: {args.num_tests}")
    print()

    # 设置采样参数 - 只生成 1 个 token 来测试预填充
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,  # 不需要 detokenize，更快
    )

    print("正在加载模型...")
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
    print(f"模型加载完成! (耗时: {load_time:.2f} 秒)")
    print()

    # 存储结果
    results = []

    for prompt_len in test_lengths:
        print("-" * 80)
        print(f"测试输入长度: {prompt_len} tokens")
        print("-" * 80)

        # 生成测试 prompt
        prompt_tokens = generate_random_prompt(tokenizer, prompt_len)
        prompts = [TokensPrompt(prompt_token_ids=prompt_tokens)]

        # 热身
        print(f"热身 ({args.num_warmup} 次)...")
        for i in range(args.num_warmup):
            try:
                llm.generate(prompts, sampling_params)
                print(f"  热身 {i+1}/{args.num_warmup} ✓")
            except Exception as e:
                print(f"  热身 {i+1}/{args.num_warmup} ✗: {e}")

        print()

        # 正式测试
        print(f"正式测试 ({args.num_tests} 次)...")
        times = []
        for i in range(args.num_tests):
            try:
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                throughput = prompt_len / elapsed
                print(f"  测试 {i+1}/{args.num_tests}: {elapsed:.4f}s, {throughput:.2f} tok/s")
            except Exception as e:
                print(f"  测试 {i+1}/{args.num_tests} ✗: {e}")

        print()

        # 计算统计数据
        if times:
            import statistics
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            avg_throughput = prompt_len / avg_time
            max_throughput = prompt_len / min_time
            min_throughput = prompt_len / max_time

            print(f"统计结果 (长度={prompt_len}):")
            print(f"  平均耗时: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"  范围: {min_time:.4f}s ~ {max_time:.4f}s")
            print(f"  平均吞吐量: {avg_throughput:.2f} tok/s")
            print(f"  最好吞吐量: {max_throughput:.2f} tok/s")
            print(f"  最差吞吐量: {min_throughput:.2f} tok/s")

            results.append({
                "prompt_len": prompt_len,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_time": std_time,
                "avg_throughput": avg_throughput,
                "max_throughput": max_throughput,
                "min_throughput": min_throughput,
            })

        print()

    # 输出汇总表格
    print("=" * 80)
    print("压测结果汇总")
    print("=" * 80)
    print()
    print(f"{'长度(tok)':<12} {'平均(s)':<10} {'±std(s)':<10} {'平均(tok/s)':<15} {'最好(tok/s)':<15}")
    print("-" * 72)
    for r in results:
        print(f"{r['prompt_len']:<12} {r['avg_time']:<10.4f} {r['std_time']:<10.4f} {r['avg_throughput']:<15.2f} {r['max_throughput']:<15.2f}")

    print()
    print("=" * 80)
    print("压测完成!")
    print("=" * 80)

    # 保存结果到 JSON
    import json
    output_file = "prefill_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print()
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
