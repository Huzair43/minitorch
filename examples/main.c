#include <stdio.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"
#include "minitorch/core/tensor_linalg.h"

int main() {
    printf("=== MiniTorch v2 - Examples ===\n\n");
    
    /* --- 1D Vectors --- */
    printf("1. Vecteurs 1D (shape=[3])\n");
    int shape_1d[1] = {3};
    Tensor* a = tensor_create(shape_1d, 1);
    Tensor* b = tensor_create(shape_1d, 1);
    
    tensor_set(a, (int[]){0}, 1.0f);
    tensor_set(a, (int[]){1}, 2.0f);
    tensor_set(a, (int[]){2}, 3.0f);
    
    tensor_set(b, (int[]){0}, 4.0f);
    tensor_set(b, (int[]){1}, 5.0f);
    tensor_set(b, (int[]){2}, 6.0f);
    
    printf("a = ");
    tensor_print(a);
    printf("b = ");
    tensor_print(b);
    
    Tensor* c = tensor_add(a, b);
    printf("a + b = ");
    tensor_print(c);
    
    Tensor* d = tensor_mul(a, b);
    printf("a * b (elementwise) = ");
    tensor_print(d);
    
    tensor_free(c);
    tensor_free(d);
    tensor_free(a);
    tensor_free(b);
    
    /* --- 2D Matrices --- */
    printf("\n2. Matrices 2D (shape=[2, 3])\n");
    int shape_2d[2] = {2, 3};
    Tensor* mat_a = tensor_create(shape_2d, 2);
    
    tensor_set(mat_a, (int[]){0, 0}, 1.0f);
    tensor_set(mat_a, (int[]){0, 1}, 2.0f);
    tensor_set(mat_a, (int[]){0, 2}, 3.0f);
    tensor_set(mat_a, (int[]){1, 0}, 4.0f);
    tensor_set(mat_a, (int[]){1, 1}, 5.0f);
    tensor_set(mat_a, (int[]){1, 2}, 6.0f);
    
    printf("Matrix A (2x3) = ");
    tensor_print(mat_a);
    
    /* --- Matrix Multiplication --- */
    printf("\n3. Multiplication matricielle (2x3) x (3x2)\n");
    int shape_mat_b[2] = {3, 2};
    Tensor* mat_b = tensor_create(shape_mat_b, 2);
    
    tensor_set(mat_b, (int[]){0, 0}, 1.0f);
    tensor_set(mat_b, (int[]){0, 1}, 2.0f);
    tensor_set(mat_b, (int[]){1, 0}, 3.0f);
    tensor_set(mat_b, (int[]){1, 1}, 4.0f);
    tensor_set(mat_b, (int[]){2, 0}, 5.0f);
    tensor_set(mat_b, (int[]){2, 1}, 6.0f);
    
    printf("Matrix B (3x2) = ");
    tensor_print(mat_b);
    
    Tensor* mat_c = tensor_matmul(mat_a, mat_b);
    printf("A @ B (2x2) = ");
    tensor_print(mat_c);
    
    tensor_free(mat_c);
    tensor_free(mat_a);
    tensor_free(mat_b);
    
    /* --- Reshape --- */
    printf("\n4. Reshape: (2x3) -> (3x2)\n");
    int shape_reshape_from[2] = {2, 3};
    Tensor* reshape_from = tensor_create(shape_reshape_from, 2);
    
    for (int i = 0; i < 6; i++) {
        int idx = i / 3;
        int jdx = i % 3;
        tensor_set(reshape_from, (int[]){idx, jdx}, (float)(i + 1));
    }
    
    printf("Original (2x3) = ");
    tensor_print(reshape_from);
    
    int shape_reshape_to[2] = {3, 2};
    Tensor* reshaped = tensor_reshape(reshape_from, shape_reshape_to, 2);
    printf("Reshaped (3x2) = ");
    tensor_print(reshaped);
    
    tensor_free(reshape_from);
    tensor_free(reshaped);
    
    /* --- Tensor Creation Utilities --- */
    printf("\n5. Tensor utilities\n");
    int shape_ones[2] = {2, 2};
    Tensor* ones = tensor_ones(shape_ones, 2);
    printf("Ones (2x2) = ");
    tensor_print(ones);
    
    Tensor* zeros = tensor_zeros(shape_ones, 2);
    printf("Zeros (2x2) = ");
    tensor_print(zeros);
    
    Tensor* cloned = tensor_clone(ones);
    printf("Clone of ones = ");
    tensor_print(cloned);
    
    tensor_free(ones);
    tensor_free(zeros);
    tensor_free(cloned);
    
    printf("\n=== Tests completed successfully! ===\n");
    
    return 0;
}
