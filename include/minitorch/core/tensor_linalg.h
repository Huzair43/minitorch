#ifndef MINITORCH_TENSOR_LINALG_H
#define MINITORCH_TENSOR_LINALG_H

#include "tensor.h"

Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
float tensor_det(const Tensor* t);
Tensor* tensor_transpose(Tensor* t, int axis1, int axis2);

#endif
