# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re

from torch.utils.cpp_extension import load


def cuda_arch_list() -> str:
    return os.environ.get("VPA_PROTENIX_CUDA_ARCH_LIST") or os.environ.get(
        "TORCH_CUDA_ARCH_LIST",
        "8.6;8.9;9.0;10.0",
    )


def extension_suffix() -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", cuda_arch_list()).strip("_")


def gencode_flags() -> list[str]:
    flags: list[str] = []
    for raw_arch in cuda_arch_list().split(";"):
        arch = raw_arch.strip()
        if not arch:
            continue
        arch = arch.removesuffix("+PTX")
        compact = arch.replace(".", "")
        flags.extend(["-gencode", f"arch=compute_{compact},code=sm_{compact}"])
    return flags


def compile(name, sources, extra_include_paths, build_directory):
    # Respect externally provided arch list; default to common VPA GPU targets.
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list()
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-std=c++17",
            "-maxrregcount=50",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            *gencode_flags(),
        ],
        verbose=True,
        build_directory=build_directory,
    )
