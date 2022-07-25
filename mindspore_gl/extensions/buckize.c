/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// return the index of a value in a list by Binary Search.
// If the value is not in the list, return the index of the next value greater than this value.
int64_t lower_bound(int64_t *A, int64_t L, int64_t R, int64_t val) {
    int64_t l = L, r = R;
    while (l < r) {
        int64_t m = (l + r) / 2;
        if (val <= A[m]) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return val <= A[l] ? l: l + 1;
}

// convert python type list to C type list
int64_t* python_Ctypes(PyObject *data_list, int64_t length) {
    int64_t* p_data_list;
    p_data_list = malloc(sizeof(int64_t) * length);
    for (int64_t index = 0; index < length; index++) {
        PyObject *item;
        item = PyList_GetItem(data_list, index);
        if (!PyFloat_Check(item))
            p_data_list[index] = 0.0;
        p_data_list[index] = PyFloat_AsDouble(item);
    }
    return p_data_list;
}

// bucket function
static PyObject *method_bucket(PyObject *self, PyObject *args) {
    PyObject *boundaries;
    PyObject *input_data;
    int64_t l;
    int64_t N;
    // Parse arguments
    if (!PyArg_ParseTuple(args, "OOLL", &boundaries, &input_data, &l, &N)) {
        return NULL;
    }
    int64_t* c_boundaries = python_Ctypes(boundaries, l);
    int64_t* c_input_data = python_Ctypes(input_data, N);
    int64_t* output_data = c_input_data;

    // call Binary Search and find the index
    for (int64_t pos = 0; pos < N; pos++) {
        int64_t bucket_idx = lower_bound(c_boundaries, 0, l, c_input_data[pos]);
        output_data[pos] = bucket_idx;
    }

    // convert C type list to python type list
    PyObject* python_val = PyList_New(N);
    for (int64_t i = 0; i < N; i++) {
        PyObject* python_int = Py_BuildValue("i", output_data[i]);
        PyList_SetItem(python_val, i, python_int);
    }
    return python_val;
}

// define the function interface of C extended modules
static PyMethodDef FputsMethods[] = {
    {"bucket", method_bucket, METH_VARARGS, "Python interface for buck C library function"},
    {NULL, NULL, 0, NULL}
};

// the structure to describe the method of C extension type
static struct PyModuleDef fputsmodule = {
    PyModuleDef_HEAD_INIT,
    "buck",
    "Python interface for the buck C library function",
    -1,
    FputsMethods
};

// create and initialize extension modules
PyMODINIT_FUNC PyInit_bucket_kernel(void) {
    PyObject *module = PyModule_Create(&fputsmodule);
    return module;
}
