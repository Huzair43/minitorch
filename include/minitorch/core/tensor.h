#ifndef MINITORCH_TENSOR_H
#define MINITORCH_TENSOR_H

typedef struct {
    float* data;
    int size;
} Tensor;

Tensor* tensor_create(int size);
void tensor_free(Tensor* t);

Tensor* tensor_add(const Tensor* a, const Tensor* b);

void tensor_print(const Tensor* t);

#endif