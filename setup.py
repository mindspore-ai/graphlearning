# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""setup"""
import sys
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
###########################
# Build Cython Extension
###########################


class CustomBuildExt(_build_ext):
    """CustomBuildExt"""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


compile_extra_args = ["-std=c++11", "-O3", "-fopenmp"]

link_extra_args = ["-fopenmp"]
if sys.platform.startswith("darwin"):
    compile_extra_args = ['-std=c++11', "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

############################
# Check Platform
############################

assert sys.platform.startswith("linux") or sys.platform.startswith("darwin"), "Only Supported On Linux And Darwin"

CYTHON_BUCKET_KERNEL = Extension(
    "mindspore_gl.bucket_kernel",
    [
        "mindspore_gl/extensions/buckize.c",
    ]
)

CYTHON_SAMPLE_KERNEL = Extension(
    "mindspore_gl.sample_kernel",
    [
        "mindspore_gl/extensions/sample_kernel.pyx",
    ],
    language="c++",
    extra_compile_args=compile_extra_args,
    extra_link_args=link_extra_args,
)

CYTHON_MEMORY_KERNEL = Extension(
    "mindspore_gl.memory_kernel",
    [
        "mindspore_gl/extensions/memory_reference_counter.pyx",
    ],
    language="c++",
    extra_compile_args=compile_extra_args,
    extra_link_args=link_extra_args,
)

CYTHON_ARRAY_KERNEL = Extension(
    "mindspore_gl.array_kernel",
    [
        "mindspore_gl/extensions/array_kernel.pyx",
    ],
    language="c++",
    extra_compile_args=compile_extra_args,
    extra_link_args=link_extra_args,
)

SHAERED_NUMPY_LINUX = Extension(
    "mindspore_gl/dataloader/shared_numpy/_posixshmem",
    define_macros=[
        ("HAVE_SHM_OPEN", "1"),
        ("HAVE_SHM_UNLINK", "1"),
        ("HAVE_SHM_MMAN_H", 1),
    ],
    libraries=["rt"],
    sources=["mindspore_gl/dataloader/shared_numpy/posixshmem.c"],
)

SHAERED_NUMPY_DARWIN = Extension(
    "mindspore_gl/dataloader/shared_numpy/_posixshmem",
    define_macros=[
        ("HAVE_SHM_OPEN", "1"),
        ("HAVE_SHM_UNLINK", "1"),
        ("HAVE_SHM_MMAN_H", 1),
    ],
    sources=["mindspore_gl/dataloader/shared_numpy/posixshmem.c"],
)
SHAERED_NUMPY = SHAERED_NUMPY_LINUX
if sys.platform.startswith("darwin"):
    SHAERED_NUMPY = SHAERED_NUMPY_DARWIN

extensions = [
    CYTHON_BUCKET_KERNEL,
    CYTHON_SAMPLE_KERNEL,
    CYTHON_MEMORY_KERNEL,
    CYTHON_ARRAY_KERNEL,
    SHAERED_NUMPY

]

setuptools.setup(
    name='mindspore-gl',
    version='0.2',
    author='The MindSpore GraphLearning Authors',
    author_email='contact@mindspore.cn',
    description='MindSpore graph learning',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'ast-decompiler>=0.3.2',
        'astpretty',
        'scikit-learn>=0.24.2',
        'Cython',
        'networkx>=2.6.3',
        'rdkit>=2022.9.1',
        'decorator>=5.1.1',
        'tqdm>=4.64.1',
        'pandas>=1.1.5',
    ],
    zip_safe=False,
    #################################
    #  Build Extension
    #################################
    cmdclass={'build_ext': CustomBuildExt},
    include_package_data=True,
    ext_modules=extensions,

)
