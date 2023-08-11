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
"""Utils function"""
import os
import stat
import importlib.util
from pathlib import Path
import random

DEFAULT_SRC_LOC = Path.cwd().parent / ".mindspore_gl" / str(hash(random.randint(1, 1e30)))
counter = 0

def src_to_function(src_code: str, func_name: str, globals_dict: dict, translate_path: None or str):
    """
    Transform the source code to function.

    Args:
        src_code (str): source code.
        func_name (str): the function name.
        globals_dict (dict): globals dict.
        translate_path (None or str): The path for save the construct file.

    Returns:
        Function, new function.
    """
    global counter
    module_name = func_name + f"-{counter}"
    file_name = module_name + ".py"
    counter += 1
    if translate_path is None:
        translate_path = DEFAULT_SRC_LOC
    else:
        translate_path = Path(translate_path) / ".mindspore_gl" / str(hash(random.randint(1, 1e30)))
    Path(translate_path).mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(translate_path / file_name, flags, modes), 'w') as f:
        f.write(src_code)

    new_fn = import_func(module_name, func_name, translate_path)
    return add_import_info(new_fn, globals_dict)

def add_import_info(new_fn, globals_dict):
    """
    Add import information.

    Args:
        new_fn (Function): function from source.
        globals_dict (dict): globals dict.

    Returns:
        Function, function after add the globals.
    """
    for k, v in globals_dict.items():
        if k not in new_fn.__globals__:
            new_fn.__globals__[k] = v
    return new_fn


def import_func(module_name: str,
                func_name: str,
                src_folder: Path = DEFAULT_SRC_LOC):
    """Import the function."""
    spec = importlib.util.spec_from_file_location(
        module_name, src_folder / (module_name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__dict__[func_name]
