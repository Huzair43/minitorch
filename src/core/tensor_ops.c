#include <stdio.h>
#include <stdlib.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"

static void tensor_flat_to_indices(int flat, const int* shape, int ndim, int* indices) {
    if (ndim <= 0) return;
    for (int axis = ndim - 1; axis >= 0; axis--) {
        indices[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
}

static int tensor_broadcast_shape(const Tensor* a, const Tensor* b, int* out_shape) {
    int out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int axis = 0; axis < out_ndim; axis++) {
        int a_axis = a->ndim - out_ndim + axis;
        int b_axis = b->ndim - out_ndim + axis;
        int a_dim = a_axis >= 0 ? a->shape[a_axis] : 1;
        int b_dim = b_axis >= 0 ? b->shape[b_axis] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return -1;
        }
        out_shape[axis] = a_dim > b_dim ? a_dim : b_dim;
    }
    return out_ndim;
}

static int tensor_broadcast_index(const Tensor* t, const int* out_indices, int out_ndim, int* scratch) {
    int shift = out_ndim - t->ndim;
    for (int axis = 0; axis < t->ndim; axis++) {
        int out_axis = axis + shift;
        if (out_axis < 0) {
            scratch[axis] = 0;
        } else if (t->shape[axis] == 1) {
            scratch[axis] = 0;
        } else {
            scratch[axis] = out_indices[out_axis];
        }
    }
    return tensor_get_index(t, scratch);
}

static Tensor* tensor_binary_broadcast(const Tensor* a, const Tensor* b, float (*op)(float, float)) {
    if (!a || !b) return NULL;

    int out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int* out_shape_buf = out_ndim > 0 ? (int*)malloc(sizeof(int) * (size_t)out_ndim) : NULL;
    if (out_ndim > 0 && !out_shape_buf) return NULL;

    int computed_ndim = tensor_broadcast_shape(a, b, out_shape_buf);
    if (computed_ndim < 0) {
        free(out_shape_buf);
        return NULL;
    }

    out_ndim = computed_ndim;

    Tensor* out = tensor_create(out_shape_buf, out_ndim);
    free(out_shape_buf);
    if (!out) return NULL;

    int* out_indices = out_ndim > 0 ? (int*)calloc((size_t)out_ndim, sizeof(int)) : NULL;
    int* a_indices = a->ndim > 0 ? (int*)calloc((size_t)a->ndim, sizeof(int)) : NULL;
    int* b_indices = b->ndim > 0 ? (int*)calloc((size_t)b->ndim, sizeof(int)) : NULL;
    if ((out_ndim > 0 && !out_indices) || (a->ndim > 0 && !a_indices) || (b->ndim > 0 && !b_indices)) {
        free(out_indices);
        free(a_indices);
        free(b_indices);
        tensor_free(out);
        return NULL;
    }

    for (int flat = 0; flat < out->size; flat++) {
        tensor_flat_to_indices(flat, out->shape, out_ndim, out_indices);
        int a_idx = tensor_broadcast_index(a, out_indices, out_ndim, a_indices);
        int b_idx = tensor_broadcast_index(b, out_indices, out_ndim, b_indices);
        out->data[flat] = op(a->data[a_idx], b->data[b_idx]);
    }

    free(out_indices);
    free(a_indices);
    free(b_indices);
    return out;
}

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    return tensor_binary_broadcast(a, b, op_add);
}

Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    return tensor_binary_broadcast(a, b, op_sub);
}

Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    return tensor_binary_broadcast(a, b, op_mul);
}

Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    int out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int* out_shape_buf = out_ndim > 0 ? (int*)malloc(sizeof(int) * (size_t)out_ndim) : NULL;
    if (out_ndim > 0 && !out_shape_buf) return NULL;

    int computed_ndim = tensor_broadcast_shape(a, b, out_shape_buf);
    if (computed_ndim < 0) {
        free(out_shape_buf);
        return NULL;
    }
    out_ndim = computed_ndim;

    Tensor* out = tensor_create(out_shape_buf, out_ndim);
    free(out_shape_buf);
    if (!out) return NULL;

    int* out_indices = out_ndim > 0 ? (int*)calloc((size_t)out_ndim, sizeof(int)) : NULL;
    int* a_indices = a->ndim > 0 ? (int*)calloc((size_t)a->ndim, sizeof(int)) : NULL;
    int* b_indices = b->ndim > 0 ? (int*)calloc((size_t)b->ndim, sizeof(int)) : NULL;
    if ((out_ndim > 0 && !out_indices) || (a->ndim > 0 && !a_indices) || (b->ndim > 0 && !b_indices)) {
        free(out_indices);
        free(a_indices);
        free(b_indices);
        tensor_free(out);
        return NULL;
    }

    for (int flat = 0; flat < out->size; flat++) {
        tensor_flat_to_indices(flat, out->shape, out_ndim, out_indices);
        int a_idx = tensor_broadcast_index(a, out_indices, out_ndim, a_indices);
        int b_idx = tensor_broadcast_index(b, out_indices, out_ndim, b_indices);
        if (b->data[b_idx] == 0.0f) {
            printf("Erreur: Division par zero a l'indice %d\n", flat);
            free(out_indices);
            free(a_indices);
            free(b_indices);
            tensor_free(out);
            return NULL;
        }
        out->data[flat] = a->data[a_idx] / b->data[b_idx];
    }

    free(out_indices);
    free(a_indices);
    free(b_indices);
    return out;
}

Tensor* tensor_sum(const Tensor* t, int axis) {
    if (!t) return NULL;
    if (axis < 0 || axis >= t->ndim) {
        printf("Erreur: axis %d invalide (ndim=%d)\n", axis, t->ndim);
        return NULL;
    }

    int new_ndim = t->ndim - 1;
    int* new_shape = new_ndim > 0 ? (int*)malloc(sizeof(int) * (size_t)new_ndim) : NULL;
    if (new_ndim > 0 && !new_shape) return NULL;

    for (int i = 0, j = 0; i < t->ndim; i++) {
        if (i != axis) {
            new_shape[j++] = t->shape[i];
        }
    }

    Tensor* result = tensor_create(new_shape, new_ndim);
    free(new_shape);
    if (!result) return NULL;

    int* out_indices = new_ndim > 0 ? (int*)calloc((size_t)new_ndim, sizeof(int)) : NULL;
    int* in_indices = (int*)calloc((size_t)t->ndim, sizeof(int));
    if ((new_ndim > 0 && !out_indices) || !in_indices) {
        free(out_indices);
        free(in_indices);
        tensor_free(result);
        return NULL;
    }

    for (int flat = 0; flat < result->size; flat++) {
        if (new_ndim > 0) {
            tensor_flat_to_indices(flat, result->shape, new_ndim, out_indices);
        }

        for (int i = 0, j = 0; i < t->ndim; i++) {
            if (i == axis) {
                continue;
            }
            in_indices[i] = (new_ndim > 0) ? out_indices[j++] : 0;
        }

        float sum = 0.0f;
        for (int red = 0; red < t->shape[axis]; red++) {
            in_indices[axis] = red;
            sum += t->data[tensor_get_index(t, in_indices)];
        }
        result->data[flat] = sum;
    }

    free(out_indices);
    free(in_indices);
    return result;
}

Tensor* tensor_mean(const Tensor* t, int axis) {
    Tensor* sum = tensor_sum(t, axis);
    if (!sum) return NULL;

    float divisor = (float)t->shape[axis];
    for (int i = 0; i < sum->size; i++) {
        sum->data[i] /= divisor;
    }

    return sum;
}
