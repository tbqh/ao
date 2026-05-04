# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax
from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)

# Soft dependency on TransformerEngine
try:
    from transformer_engine.pytorch.tensor import NVFP4Quantizer
    HAS_TE = True
except ImportError:
    HAS_TE = False

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
N_SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int


@dataclass(frozen=True)
class ExperimentResult:
    triton_time_us: float
    triton_gbps: float
    te_time_us: Optional[float] = None
    te_gbps: Optional[float] = None


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def run_experiment(config: ExperimentConfig) -> ExperimentResult | None:
    m, n = config.m, config.n
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)

    # Triton benchmark (amax + quantize together for fair comparison with TE)
    def triton_full_quantize(x):
        col_amax, row_amax = triton_rht_amax(x)
        return triton_rht_quantize_row_col(x, col_global_amax=col_amax, row_global_amax=row_amax)

    try:
        triton_full_quantize(x)  # Warmup / check if implemented
        triton_time_us = benchmark_cuda_function_in_microseconds(triton_full_quantize, x)
    except NotImplementedError:
        return None

    read_bytes = m * n * 2  # bfloat16 input
    col_write = n * (m // 2) + (n // 128) * (m // 64) * 32 * 16
    row_write = m * (n // 2) + (m // 128) * (n // 64) * 32 * 16
    total_bytes = read_bytes + col_write + row_write
    triton_gbps = (total_bytes / 1e9) / (triton_time_us / 1e6)

    # TE benchmark (if available)
    te_time_us = None
    te_gbps = None
    if HAS_TE:
        te_quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=False,
            with_rht=True,
            with_post_rht_amax=True,
        )
        te_time_us = benchmark_cuda_function_in_microseconds(te_quantizer, x)
        te_gbps = (total_bytes / 1e9) / (te_time_us / 1e6)

    return ExperimentResult(
        triton_time_us=triton_time_us,
        triton_gbps=triton_gbps,
        te_time_us=te_time_us,
        te_gbps=te_gbps,
    )


def print_results(experiments: List[Experiment]):
    if HAS_TE:
        headers = ["M", "N", "Triton us", "Triton GB/s", "TE us", "TE GB/s", "Speedup"]
        rows = [
            [
                e.config.m,
                e.config.n,
                round(e.result.triton_time_us, 3),
                round(e.result.triton_gbps, 3),
                round(e.result.te_time_us, 3) if e.result.te_time_us else "N/A",
                round(e.result.te_gbps, 3) if e.result.te_gbps else "N/A",
                f"{e.result.te_time_us / e.result.triton_time_us:.2f}x" if e.result.te_time_us else "N/A",
            ]
            for e in experiments
        ]
    else:
        headers = ["M", "N", "Triton us", "Triton GB/s"]
        rows = [
            [
                e.config.m,
                e.config.n,
                round(e.result.triton_time_us, 3),
                round(e.result.triton_gbps, 3),
            ]
            for e in experiments
        ]
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
