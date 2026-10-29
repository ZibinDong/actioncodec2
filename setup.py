"""Build the ActionCodec2 physical-stage kernels; metadata lives in pyproject.toml.

``ACTIONCODEC2_NO_NATIVE=1`` skips the extension. ActionCodec2 then refuses to
load unless the same variable is set at run time, which selects the slower
NumPy path bundled with the artifact.
"""

import os
import sys

from setuptools import setup

if os.environ.get("ACTIONCODEC2_NO_NATIVE"):
    setup()
else:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    native = "src/actioncodec2/tokenizers/native"
    shared_native = "src/actioncodec2/core/backends/actioncodec2/native"
    training_openmp = ["-fopenmp"] if sys.platform.startswith("linux") else []
    setup(
        ext_modules=[
            Pybind11Extension(
                "actioncodec2.tokenizers._actioncodec2_native",
                sources=[f"{native}/{name}.cpp" for name in (
                    "module", "physical", "so3", "setbpe", "resample", "reconstruct", "grammar"
                )],
                include_dirs=[native],
                depends=[f"{native}/{name}.hpp" for name in ("parallel", "quaternion")],
                # No -march=native or fast-math: results must match the NumPy path bit for bit.
                extra_compile_args=["-O3", "-fvisibility=hidden", "-ffp-contract=off"],
                cxx_std=17,
            ),
            Pybind11Extension(
                "actioncodec2.core.backends.actioncodec2._native",
                sources=[f"{shared_native}/{name}.cpp" for name in (
                    "module", "physical", "so3", "resample", "inference", "preprocessing", "sparse", "spatial", "temporal", "bpe", "setbpe", "grammar", "replay"
                )],
                include_dirs=[shared_native],
                depends=[f"{shared_native}/{name}.hpp" for name in ("parallel", "quaternion", "common", "token_streams", "repacking")],
                extra_compile_args=["-O3", "-fvisibility=hidden", "-ffp-contract=off", *training_openmp],
                extra_link_args=training_openmp,
                cxx_std=17,
            ),
        ],
        cmdclass={"build_ext": build_ext},
    )
