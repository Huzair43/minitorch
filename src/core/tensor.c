#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minitorch/core/tensor.h"

Tensor* tensor_create(const int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    
    t->shape = (int*)malloc(sizeof(int) * ndim);
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }

    t->strides = (int*)malloc(sizeof(int) * ndim);
    
    int current_size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = current_size;
        current_size *= shape[i];
    }
    t->size = current_size;

    t->data = (float*)calloc(t->size, sizeof(float)); 
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
    
    memcpy(clone->data, t->data, t->size * sizeof(float));
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
    int index = 0;
    for (int i = 0; i < t->ndim; i++) {
        index += indices[i] * t->strides[i];
    }
    return index;
}

float tensor_get(const Tensor* t, const int* indices) {
    int idx = tensor_get_index(t, indices);
    return t->data[idx];
}

void tensor_set(Tensor* t, const int* indices, float value) {
    int idx = tensor_get_index(t, indices);
    t->data[idx] = value;
}

Tensor* tensor_reshape(const Tensor* t, const int* new_shape, int new_ndim) {
    if (!t) return NULL;
    
    int new_size = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }
    
    if (new_size != t->size) {
        printf("Erreur: reshape incompatible. Ancien size: %d, nouveau: %d\n", t->size, new_size);
        return NULL;
    }
    
    Tensor* reshaped = tensor_create(new_shape, new_ndim);
    if (!reshaped) return NULL;
    
    memcpy(reshaped->data, t->data, t->size * sizeof(float));
    return reshaped;
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
        printf("[]\n");
        return;
    }
    
    int* temp_indices = (int*)calloc(t->ndim, sizeof(int));
    _tensor_print_recursive(t, 0, temp_indices);
    printf("\n");
    free(temp_indices);
}
