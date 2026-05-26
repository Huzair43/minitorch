#include <stdio.h>
#include "minitorch/core/tensor.h"

int main() {
    Tensor* a = tensor_create(3);
    Tensor* b = tensor_create(3);

    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    a->data[2] = 3.0f;

    b->data[0] = 4.0f;
    b->data[1] = 5.0f;
    b->data[2] = 6.0f;

    Tensor* c = tensor_add(a, b);

    tensor_print(a);
    tensor_print(b);
    tensor_print(c);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);

    return 0;
}