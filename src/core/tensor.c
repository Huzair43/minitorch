#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "minitorch/core/tensor.h"

int tensor_numel(const int* shape, int ndim) {
    if (ndim < 0) return 0;
    if (ndim == 0) return 1;
    if (!shape) return 0;

    long long size = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return 0;
        if (size > LLONG_MAX / shape[i]) return 0;
        size *= shape[i];
        if (size > INT_MAX) return 0;
    }
    return (int)size;
}

int tensor_same_shape(const Tensor* a, const Tensor* b) {
    if (!a || !b) return 0;
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

Tensor* tensor_create(const int* shape, int ndim) {
    if (ndim < 0) return NULL;
    if (ndim > 0 && !shape) return NULL;

    int total = tensor_numel(shape, ndim);
    if (total <= 0) return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->size = total;
    t->shape = ndim > 0 ? (int*)malloc(sizeof(int) * (size_t)ndim) : NULL;
    t->strides = ndim > 0 ? (int*)malloc(sizeof(int) * (size_t)ndim) : NULL;

    if (ndim > 0 && (!t->shape || !t->strides)) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= shape[i];
    }

    t->data = (float*)calloc((size_t)t->size, sizeof(float));
    if (!t->data) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    return t;
}

Tensor* tensor_zeros(const int* shape, int ndim) {
    return tensor_create(shape, ndim);
}

Tensor* tensor_ones(const int* shape, int ndim) {
    Tensor* t = tensor_create(shape, ndim);
    if (!t) return NULL;

    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f;
    }
    return t;
}

Tensor* tensor_clone(const Tensor* t) {
    if (!t) return NULL;

    Tensor* clone = tensor_create(t->shape, t->ndim);
    if (!clone) return NULL;

    memcpy(clone->data, t->data, sizeof(float) * (size_t)t->size);
    return clone;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

int tensor_get_index(const Tensor* t, const int* indices) {
    if (!t) return -1;
    if (t->ndim == 0) return 0;
    if (!indices) return -1;

    int index = 0;
    for (int i = 0; i < t->ndim; i++) {
        index += indices[i] * t->strides[i];
    }
    return index;
}

float tensor_get(const Tensor* t, const int* indices) {
    int idx = tensor_get_index(t, indices);
    if (idx < 0) return 0.0f;
    return t->data[idx];
}

void tensor_set(Tensor* t, const int* indices, float value) {
    int idx = tensor_get_index(t, indices);
    if (idx < 0) return;
    t->data[idx] = value;
}

Tensor* tensor_reshape(const Tensor* t, const int* new_shape, int new_ndim) {
    if (!t) return NULL;

    int new_size = tensor_numel(new_shape, new_ndim);
    if (new_size != t->size) {
        printf("Erreur: reshape incompatible. Ancien size: %d, nouveau: %d\n", t->size, new_size);
        return NULL;
    }

    Tensor* reshaped = tensor_create(new_shape, new_ndim);
    if (!reshaped) return NULL;

    memcpy(reshaped->data, t->data, sizeof(float) * (size_t)t->size);
    return reshaped;
}

Tensor* tensor_broadcast_to(const Tensor* t, const int* target_shape, int target_ndim) {
    if (!t) return NULL;
    if (target_ndim < t->ndim) return NULL;
    if (target_ndim > 0 && !target_shape) return NULL;

    int target_size = tensor_numel(target_shape, target_ndim);
    if (target_size <= 0) return NULL;

    Tensor* out = tensor_create(target_shape, target_ndim);
    if (!out) return NULL;

    int* out_indices = target_ndim > 0 ? (int*)calloc((size_t)target_ndim, sizeof(int)) : NULL;
    int* in_indices = t->ndim > 0 ? (int*)calloc((size_t)t->ndim, sizeof(int)) : NULL;
    if ((target_ndim > 0 && !out_indices) || (t->ndim > 0 && !in_indices)) {
        free(out_indices);
        free(in_indices);
        tensor_free(out);
        return NULL;
    }

    int shift = target_ndim - t->ndim;
    for (int flat = 0; flat < target_size; flat++) {
        int remainder = flat;
        for (int axis = target_ndim - 1; axis >= 0; axis--) {
            out_indices[axis] = remainder % target_shape[axis];
            remainder /= target_shape[axis];
        }

        for (int axis = 0; axis < t->ndim; axis++) {
            int out_axis = axis + shift;
            if (out_axis < 0) {
                in_indices[axis] = 0;
                continue;
            }

            if (t->shape[axis] != 1 && t->shape[axis] != target_shape[out_axis]) {
                free(out_indices);
                free(in_indices);
                tensor_free(out);
                return NULL;
            }
            in_indices[axis] = (t->shape[axis] == 1) ? 0 : out_indices[out_axis];
        }

        int out_idx = tensor_get_index(out, out_indices);
        int in_idx = tensor_get_index(t, in_indices);
        out->data[out_idx] = t->data[in_idx];
    }

    free(out_indices);
    free(in_indices);
    return out;
}

void _tensor_print_recursive(const Tensor* t, int current_dim, int* current_indices) {
    if (current_dim == t->ndim) {
        int idx = tensor_get_index(t, current_indices);
        printf("%f", t->data[idx]);
        return;
    }

    printf("[");
    for (int i = 0; i < t->shape[current_dim]; i++) {
        current_indices[current_dim] = i;
        _tensor_print_recursive(t, current_dim + 1, current_indices);
        if (i < t->shape[current_dim] - 1) {
            printf(", ");
            if (current_dim == t->ndim - 2) printf("\n  ");
        }
    }
    printf("]");
}

void tensor_print(const Tensor* t) {
    if (!t) {
        printf("NULL\n");
        return;
    }

    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("])\n");

    if (t->ndim == 0) {
        printf("%f\n", t->data[0]);
        return;
    }

    int* temp_indices = (int*)calloc((size_t)t->ndim, sizeof(int));
    if (!temp_indices) {
        printf("<allocation failed>\n");
        return;
    }
    _tensor_print_recursive(t, 0, temp_indices);
    printf("\n");
    free(temp_indices);
}
