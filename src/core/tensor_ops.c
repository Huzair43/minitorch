#include <stdio.h>
#include <stdlib.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL; 
    }

    Tensor* out = tensor_create(a->shape, a->ndim);
    if (!out) return NULL;

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return out;
}

Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL; 
    }

    Tensor* out = tensor_create(a->shape, a->ndim);
    if (!out) return NULL;

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return out;
}

Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL; 
    }

    Tensor* out = tensor_create(a->shape, a->ndim);
    if (!out) return NULL;

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return out;
}

Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL; 
    }

    Tensor* out = tensor_create(a->shape, a->ndim);
    if (!out) return NULL;

    for (int i = 0; i < a->size; i++) {
        if (b->data[i] == 0.0f) {
            printf("Erreur: Division par zero a l'indice %d\n", i);
            tensor_free(out);
            return NULL;
        }
        out->data[i] = a->data[i] / b->data[i];
    }

    return out;
}

Tensor* tensor_sum(const Tensor* t, int axis) {
    if (!t) return NULL;
    if (axis < 0 || axis >= t->ndim) {
        printf("Erreur: axis %d invalide (ndim=%d)\n", axis, t->ndim);
        return NULL;
    }
    
    int new_ndim = t->ndim - 1;
    int new_shape[10];
    int shape_idx = 0;
    
    for (int i = 0; i < t->ndim; i++) {
        if (i != axis) {
            new_shape[shape_idx++] = t->shape[i];
        }
    }
    
    Tensor* result = tensor_create(new_shape, new_ndim);
    if (!result) return NULL;
    
    int* indices = (int*)calloc(t->ndim, sizeof(int));
    for (int i = 0; i < result->size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < t->shape[axis]; j++) {
            int result_idx = 0;
            int result_stride_idx = 0;
            
            for (int k = 0; k < t->ndim; k++) {
                if (k == axis) {
                    indices[k] = j;
                } else {
                    int dim_val = (i / result->strides[result_stride_idx]) % result->shape[result_stride_idx];
                    indices[k] = dim_val;
                    result_stride_idx++;
                }
            }
            
            int idx = tensor_get_index(t, indices);
            sum += t->data[idx];
        }
        result->data[i] = sum;
    }
    
    free(indices);
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
