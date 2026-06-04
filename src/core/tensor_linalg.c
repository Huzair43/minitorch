#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_linalg.h"

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        printf("Erreur : matmul supporte uniquement les tenseurs 2D.\n");
        return NULL;
    }
    if (a->shape[1] != b->shape[0]) {
        printf("Erreur dimensions incompatibles pour matmul : [%d, %d] x [%d, %d]\n", 
               a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        return NULL;
    }

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    int out_shape[2] = {M, N};
    Tensor* out = tensor_create(out_shape, 2);
    if (!out) return NULL;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int idx_a = i * a->strides[0] + k * a->strides[1];
                int idx_b = k * b->strides[0] + j * b->strides[1];
                sum += a->data[idx_a] * b->data[idx_b];
            }
            int idx_out = i * out->strides[0] + j * out->strides[1];
            out->data[idx_out] = sum;
        }
    }

    return out;
}

float tensor_det(const Tensor* t) {
    if (!t) return 0.0f;
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) {
        printf("Erreur : Le déterminant requiert une matrice carrée 2D.\n");
        return 0.0f;
    }

    int n = t->shape[0];
    
    float* lu = (float*)malloc(n * n * sizeof(float));
    if (!lu) return 0.0f;
    
    for (int i = 0; i < n * n; i++) lu[i] = t->data[i];

    float det = 1.0f;
    int permutation_count = 0;

    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (fabs(lu[j * n + i]) > fabs(lu[pivot * n + i])) {
                pivot = j;
            }
        }

        if (pivot != i) {
            for (int k = 0; k < n; k++) {
                float temp = lu[i * n + k];
                lu[i * n + k] = lu[pivot * n + k];
                lu[pivot * n + k] = temp;
            }
            permutation_count++;
        }

        if (fabs(lu[i * n + i]) < 1e-7f) {
            free(lu);
            return 0.0f;
        }

        for (int j = i + 1; j < n; j++) {
            float factor = lu[j * n + i] / lu[i * n + i];
            lu[j * n + i] = factor; 
            for (int k = i + 1; k < n; k++) {
                lu[j * n + k] -= factor * lu[i * n + k]; 
            }
        }
    }

    for (int i = 0; i < n; i++) {
        det *= lu[i * n + i];
    }

    if (permutation_count % 2 != 0) {
        det = -det;
    }

    free(lu);
    return det;
}

Tensor* tensor_transpose(Tensor* t, int axis1, int axis2) {
    if (!t) return NULL;
    if (axis1 < 0 || axis1 >= t->ndim || axis2 < 0 || axis2 >= t->ndim) {
        printf("Erreur: axes invalides pour transpose\n");
        return NULL;
    }
    
    if (axis1 == axis2) {
        return tensor_clone(t);
    }

    int* new_shape = (int*)malloc(sizeof(int) * t->ndim);
    if (!new_shape) return NULL;

    for (int i = 0; i < t->ndim; i++) {
        new_shape[i] = t->shape[i];
    }
    int temp = new_shape[axis1];
    new_shape[axis1] = new_shape[axis2];
    new_shape[axis2] = temp;

    Tensor* result = tensor_create(new_shape, t->ndim);
    free(new_shape);
    if (!result) return NULL;

    int* indices = (int*)calloc(t->ndim, sizeof(int));
    int* transposed_indices = (int*)calloc(t->ndim, sizeof(int));
    if (!indices || !transposed_indices) {
        free(indices);
        free(transposed_indices);
        tensor_free(result);
        return NULL;
    }

    for (int i = 0; i < t->size; i++) {
        int idx = i;
        for (int j = t->ndim - 1; j >= 0; j--) {
            indices[j] = idx % t->shape[j];
            idx /= t->shape[j];
        }

        for (int j = 0; j < t->ndim; j++) {
            transposed_indices[j] = indices[j];
        }

        int temp_idx = transposed_indices[axis1];
        transposed_indices[axis1] = transposed_indices[axis2];
        transposed_indices[axis2] = temp_idx;

        int out_idx = tensor_get_index(result, transposed_indices);
        int in_idx = tensor_get_index(t, indices);
        result->data[out_idx] = t->data[in_idx];
    }

    free(indices);
    free(transposed_indices);
    return result;
}
