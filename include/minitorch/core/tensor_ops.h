#ifndef MINITORCH_TENSOR_OPS_H
#define MINITORCH_TENSOR_OPS_H

#include "tensor.h"

Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_sub(const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
Tensor* tensor_div(const Tensor* a, const Tensor* b);

Tensor* tensor_sum(const Tensor* t, int axis);
Tensor* tensor_mean(const Tensor* t, int axis);

#endif
