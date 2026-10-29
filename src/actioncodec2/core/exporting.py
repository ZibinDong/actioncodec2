"""Export the same shared runtime used by both installed tokenizers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil


def save_runtime(directory: str | Path) -> list[str]:
    """Bundle portable code and AutoProcessor metadata beside fitted assets."""
    root = Path(directory)
    source = Path(__file__).absolute().parent
    destination = root / 'runtime'
    if destination.resolve() != source:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', 'native'))
    entry = root / 'processing_actioncodec2.py'
    entry.write_text((source / 'catalog.py').read_text(encoding='utf-8'), encoding='utf-8')
    for name in ('config.json', 'processor_config.json'):
        path = root / name
        metadata = json.loads(path.read_text())
        metadata['auto_map'] = {'AutoProcessor': 'processing_actioncodec2.ActionCodec2'}
        path.write_text(json.dumps(metadata, indent=2) + '\n')
    # Reuse the installed native runtime when available; a standalone artifact can build it.
    kernels = root / 'actioncodec2_kernels'
    if kernels.exists():
        shutil.rmtree(kernels)
    native_source = source / 'backends/actioncodec2/native'
    if not native_source.is_dir():
        native_source = source.parent / 'actioncodec2_kernels'
    shutil.copytree(native_source, kernels,
                    ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.o'))
    (kernels / '__init__.py').write_text('')
    (root / 'setup.py').write_text('''import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pathlib import Path
native = 'actioncodec2_kernels'
openmp = ['-fopenmp'] if sys.platform.startswith('linux') else []
setup(name='actioncodec2-kernels', version='0.1.0', packages=['actioncodec2_kernels'],
      ext_modules=[Pybind11Extension('actioncodec2_kernels._native',
          sources=sorted(str(p) for p in Path(native).glob('*.cpp')),
          include_dirs=[native], cxx_std=17,
          extra_compile_args=['-O3', '-fvisibility=hidden', '-ffp-contract=off', *openmp],
          extra_link_args=openmp)], cmdclass={'build_ext': build_ext})
''')
    (root / 'pyproject.toml').write_text('[build-system]\nrequires = ["setuptools>=77", "wheel", "pybind11>=2.12"]\nbuild-backend = "setuptools.build_meta"\n')
    license_path = next((p / 'LICENSE' for p in source.parents if (p / 'LICENSE').is_file()), None)
    if license_path is not None and license_path.resolve() != (root / 'LICENSE').resolve():
        shutil.copyfile(license_path, root / 'LICENSE')
    return [str(entry), *map(str, destination.rglob('*.py'))]
