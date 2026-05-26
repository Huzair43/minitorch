#include <stdio.h>
#include <stdlib.h>
#include "minitorch/core/tensor.h"

Tensor* tensor_create(int size) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->size = size;
    t->data = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        t->data[i] = 0.0f;
    }

    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t);
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (a->size != b->size) return NULL;

    Tensor* out = tensor_create(a->size);

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return out;
}

void tensor_print(const Tensor* t) {
    printf("[");
    for (int i = 0; i < t->size; i++) {
        printf("%f", t->data[i]);
        if (i < t->size - 1) printf(", ");
    }
    printf("]\n");
}