#!/bin/bash
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

#pip install -r requirements.txt && python setup.py bdist_wheel

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
PYTHON=$(which python3)

mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum() {
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls mindspore_gl*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

mk_new_dir "${OUTPUT_PATH}"

pip install -r ${BASEPATH}/requirements.txt --user

${PYTHON} ${BASEPATH}/setup.py bdist_wheel

mv ${BASEPATH}/dist/*whl ${OUTPUT_PATH}

write_checksum


echo "------Successfully created mindspore_gl package------"