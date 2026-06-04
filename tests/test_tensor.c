#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"
#include "minitorch/core/tensor_linalg.h"
#include "test_common.h"

/* ========== CREATION TESTS ========== */

void test_tensor_create_1d() {
    TEST_HEADER("=== Test: tensor_create_1d ===");
    
    int shape[1] = {5};
    Tensor* t = tensor_create(shape, 1);
    
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(t->ndim, 1);
    ASSERT_EQ(t->shape[0], 5);
    ASSERT_EQ(t->size, 5);
    ASSERT_NOT_NULL(t->data);
    
    tensor_free(t);
}

void test_tensor_create_2d() {
    TEST_HEADER("=== Test: tensor_create_2d ===");
    
    int shape[2] = {3, 4};
    Tensor* t = tensor_create(shape, 2);
    
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->shape[0], 3);
    ASSERT_EQ(t->shape[1], 4);
    ASSERT_EQ(t->size, 12);
    ASSERT_EQ(t->strides[0], 4);
    ASSERT_EQ(t->strides[1], 1);
    
    tensor_free(t);
}

void test_tensor_create_3d() {
    TEST_HEADER("=== Test: tensor_create_3d ===");
    
    int shape[3] = {2, 3, 4};
    Tensor* t = tensor_create(shape, 3);
    
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(t->ndim, 3);
    ASSERT_EQ(t->size, 24);
    ASSERT_EQ(t->strides[0], 12);
    ASSERT_EQ(t->strides[1], 4);
    ASSERT_EQ(t->strides[2], 1);
    
    tensor_free(t);
}

void test_tensor_zeros() {
    TEST_HEADER("=== Test: tensor_zeros ===");
    
    int shape[2] = {3, 3};
    Tensor* t = tensor_zeros(shape, 2);
    
    ASSERT_NOT_NULL(t);
    for (int i = 0; i < t->size; i++) {
        ASSERT_EQ_FLOAT(t->data[i], 0.0f, 1e-6f);
    }
    
    tensor_free(t);
}

void test_tensor_ones() {
    TEST_HEADER("=== Test: tensor_ones ===");
    
    int shape[2] = {2, 3};
    Tensor* t = tensor_ones(shape, 2);
    
    ASSERT_NOT_NULL(t);
    for (int i = 0; i < t->size; i++) {
        ASSERT_EQ_FLOAT(t->data[i], 1.0f, 1e-6f);
    }
    
    tensor_free(t);
}

void test_tensor_clone() {
    TEST_HEADER("=== Test: tensor_clone ===");
    
    int shape[2] = {2, 2};
    Tensor* t = tensor_create(shape, 2);
    for (int i = 0; i < t->size; i++) t->data[i] = (float)i;
    
    Tensor* clone = tensor_clone(t);
    
    ASSERT_NOT_NULL(clone);
    ASSERT_EQ(clone->ndim, t->ndim);
    ASSERT_EQ(clone->shape[0], t->shape[0]);
    ASSERT_EQ(clone->shape[1], t->shape[1]);
    
    for (int i = 0; i < t->size; i++) {
        ASSERT_EQ_FLOAT(clone->data[i], t->data[i], 1e-6f);
    }
    
    clone->data[0] = 999.0f;
    ASSERT_EQ_FLOAT(t->data[0], 0.0f, 1e-6f);
    
    tensor_free(t);
    tensor_free(clone);
}

/* ========== INDEXING TESTS ========== */

void test_tensor_get_index_1d() {
    TEST_HEADER("=== Test: tensor_get_index_1d ===");
    
    int shape[1] = {5};
    Tensor* t = tensor_create(shape, 1);
    
    ASSERT_EQ(tensor_get_index(t, (int[]){0}), 0);
    ASSERT_EQ(tensor_get_index(t, (int[]){2}), 2);
    ASSERT_EQ(tensor_get_index(t, (int[]){4}), 4);
    
    tensor_free(t);
}

void test_tensor_get_index_2d() {
    TEST_HEADER("=== Test: tensor_get_index_2d ===");
    
    int shape[2] = {3, 4};
    Tensor* t = tensor_create(shape, 2);
    
    ASSERT_EQ(tensor_get_index(t, (int[]){0, 0}), 0);
    ASSERT_EQ(tensor_get_index(t, (int[]){0, 1}), 1);
    ASSERT_EQ(tensor_get_index(t, (int[]){1, 0}), 4);
    ASSERT_EQ(tensor_get_index(t, (int[]){2, 3}), 11);
    
    tensor_free(t);
}

void test_tensor_get_set_values() {
    TEST_HEADER("=== Test: tensor_get_set_values ===");
    
    int shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2);
    
    tensor_set(t, (int[]){0, 0}, 1.5f);
    tensor_set(t, (int[]){1, 2}, 3.7f);
    
    ASSERT_EQ_FLOAT(tensor_get(t, (int[]){0, 0}), 1.5f, 1e-6f);
    ASSERT_EQ_FLOAT(tensor_get(t, (int[]){1, 2}), 3.7f, 1e-6f);
    ASSERT_EQ_FLOAT(tensor_get(t, (int[]){0, 1}), 0.0f, 1e-6f);
    
    tensor_free(t);
}

/* ========== RESHAPE TESTS ========== */

void test_tensor_reshape_1d_to_2d() {
    TEST_HEADER("=== Test: tensor_reshape_1d_to_2d ===");
    
    int shape_from[1] = {6};
    Tensor* t = tensor_create(shape_from, 1);
    for (int i = 0; i < 6; i++) t->data[i] = (float)(i + 1);
    
    int shape_to[2] = {2, 3};
    Tensor* reshaped = tensor_reshape(t, shape_to, 2);
    
    ASSERT_NOT_NULL(reshaped);
    ASSERT_EQ(reshaped->ndim, 2);
    ASSERT_EQ(reshaped->shape[0], 2);
    ASSERT_EQ(reshaped->shape[1], 3);
    ASSERT_EQ(reshaped->size, 6);
    
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_FLOAT(reshaped->data[i], t->data[i], 1e-6f);
    }
    
    tensor_free(t);
    tensor_free(reshaped);
}

void test_tensor_reshape_2d_to_1d() {
    TEST_HEADER("=== Test: tensor_reshape_2d_to_1d ===");
    
    int shape_from[2] = {3, 2};
    Tensor* t = tensor_create(shape_from, 2);
    for (int i = 0; i < 6; i++) t->data[i] = (float)(i);
    
    int shape_to[1] = {6};
    Tensor* reshaped = tensor_reshape(t, shape_to, 1);
    
    ASSERT_NOT_NULL(reshaped);
    ASSERT_EQ(reshaped->ndim, 1);
    ASSERT_EQ(reshaped->shape[0], 6);
    
    tensor_free(t);
    tensor_free(reshaped);
}

void test_tensor_reshape_size_mismatch() {
    TEST_HEADER("=== Test: tensor_reshape_size_mismatch ===");
    
    int shape_from[2] = {3, 2};
    Tensor* t = tensor_create(shape_from, 2);
    
    int shape_to[2] = {2, 2};
    Tensor* reshaped = tensor_reshape(t, shape_to, 2);
    
    ASSERT_NULL(reshaped);
    
    tensor_free(t);
}

/* ========== ELEMENTWISE OPERATIONS ========== */

void test_tensor_add_basic() {
    TEST_HEADER("=== Test: tensor_add_basic ===");
    
    int shape[1] = {3};
    Tensor* a = tensor_create(shape, 1);
    Tensor* b = tensor_create(shape, 1);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    b->data[0] = 4.0f; b->data[1] = 5.0f; b->data[2] = 6.0f;
    
    Tensor* c = tensor_add(a, b);
    
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_FLOAT(c->data[0], 5.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[1], 7.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[2], 9.0f, 1e-6f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_add_shape_mismatch() {
    TEST_HEADER("=== Test: tensor_add_shape_mismatch ===");
    
    int shape_a[2] = {2, 2};
    int shape_b[2] = {2, 3};
    Tensor* a = tensor_create(shape_a, 2);
    Tensor* b = tensor_create(shape_b, 2);
    
    Tensor* c = tensor_add(a, b);
    
    ASSERT_NULL(c);
    
    tensor_free(a);
    tensor_free(b);
}

void test_tensor_add_broadcast_row_vector() {
    TEST_HEADER("=== Test: tensor_add_broadcast_row_vector ===");

    int shape_a[2] = {2, 3};
    int shape_b[1] = {3};
    Tensor* a = tensor_create(shape_a, 2);
    Tensor* b = tensor_create(shape_b, 1);

    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    a->data[3] = 4.0f; a->data[4] = 5.0f; a->data[5] = 6.0f;
    b->data[0] = 10.0f; b->data[1] = 20.0f; b->data[2] = 30.0f;

    Tensor* c = tensor_add(a, b);

    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->ndim, 2);
    ASSERT_EQ(c->shape[0], 2);
    ASSERT_EQ(c->shape[1], 3);
    ASSERT_EQ_FLOAT(c->data[0], 11.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[1], 22.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[2], 33.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[3], 14.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[4], 25.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[5], 36.0f, 1e-6f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sub() {
    TEST_HEADER("=== Test: tensor_sub ===");
    
    int shape[1] = {3};
    Tensor* a = tensor_create(shape, 1);
    Tensor* b = tensor_create(shape, 1);
    
    a->data[0] = 10.0f; a->data[1] = 20.0f; a->data[2] = 30.0f;
    b->data[0] = 1.0f;  b->data[1] = 2.0f;  b->data[2] = 3.0f;
    
    Tensor* c = tensor_sub(a, b);
    
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_FLOAT(c->data[0], 9.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[1], 18.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[2], 27.0f, 1e-6f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul() {
    TEST_HEADER("=== Test: tensor_mul (elementwise) ===");
    
    int shape[2] = {2, 2};
    Tensor* a = tensor_create(shape, 2);
    Tensor* b = tensor_create(shape, 2);
    
    a->data[0] = 2.0f; a->data[1] = 3.0f; a->data[2] = 4.0f; a->data[3] = 5.0f;
    b->data[0] = 1.0f; b->data[1] = 2.0f; b->data[2] = 3.0f; b->data[3] = 4.0f;
    
    Tensor* c = tensor_mul(a, b);
    
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_FLOAT(c->data[0], 2.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[1], 6.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[2], 12.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[3], 20.0f, 1e-6f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_div() {
    TEST_HEADER("=== Test: tensor_div ===");
    
    int shape[1] = {2};
    Tensor* a = tensor_create(shape, 1);
    Tensor* b = tensor_create(shape, 1);
    
    a->data[0] = 10.0f; a->data[1] = 20.0f;
    b->data[0] = 2.0f;  b->data[1] = 4.0f;
    
    Tensor* c = tensor_div(a, b);
    
    ASSERT_NOT_NULL(c);
    ASSERT_EQ_FLOAT(c->data[0], 5.0f, 1e-6f);
    ASSERT_EQ_FLOAT(c->data[1], 5.0f, 1e-6f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sum_and_mean() {
    TEST_HEADER("=== Test: tensor_sum_and_mean ===");

    int shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2);
    for (int i = 0; i < 6; i++) t->data[i] = (float)(i + 1);

    Tensor* sum_axis0 = tensor_sum(t, 0);
    Tensor* sum_axis1 = tensor_sum(t, 1);
    Tensor* mean_axis1 = tensor_mean(t, 1);

    ASSERT_NOT_NULL(sum_axis0);
    ASSERT_NOT_NULL(sum_axis1);
    ASSERT_NOT_NULL(mean_axis1);
    ASSERT_EQ(sum_axis0->ndim, 1);
    ASSERT_EQ(sum_axis0->shape[0], 3);
    ASSERT_EQ_FLOAT(sum_axis0->data[0], 5.0f, 1e-6f);
    ASSERT_EQ_FLOAT(sum_axis0->data[1], 7.0f, 1e-6f);
    ASSERT_EQ_FLOAT(sum_axis0->data[2], 9.0f, 1e-6f);
    ASSERT_EQ_FLOAT(sum_axis1->data[0], 6.0f, 1e-6f);
    ASSERT_EQ_FLOAT(sum_axis1->data[1], 15.0f, 1e-6f);
    ASSERT_EQ_FLOAT(mean_axis1->data[0], 2.0f, 1e-6f);
    ASSERT_EQ_FLOAT(mean_axis1->data[1], 5.0f, 1e-6f);

    tensor_free(t);
    tensor_free(sum_axis0);
    tensor_free(sum_axis1);
    tensor_free(mean_axis1);
}

/* ========== LINEAR ALGEBRA TESTS ========== */

void test_tensor_matmul_basic() {
    TEST_HEADER("=== Test: tensor_matmul_basic ===");
    
    int shape_a[2] = {2, 2};
    int shape_b[2] = {2, 2};
    Tensor* a = tensor_create(shape_a, 2);
    Tensor* b = tensor_create(shape_b, 2);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;
    
    b->data[0] = 5.0f; b->data[1] = 6.0f;
    b->data[2] = 7.0f; b->data[3] = 8.0f;
    
    Tensor* c = tensor_matmul(a, b);
    
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->shape[0], 2);
    ASSERT_EQ(c->shape[1], 2);
    ASSERT_EQ_FLOAT(c->data[0], 19.0f, 1e-5f);
    ASSERT_EQ_FLOAT(c->data[1], 22.0f, 1e-5f);
    ASSERT_EQ_FLOAT(c->data[2], 43.0f, 1e-5f);
    ASSERT_EQ_FLOAT(c->data[3], 50.0f, 1e-5f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_incompatible() {
    TEST_HEADER("=== Test: tensor_matmul_incompatible ===");
    
    int shape_a[2] = {2, 3};
    int shape_b[2] = {2, 2};
    Tensor* a = tensor_create(shape_a, 2);
    Tensor* b = tensor_create(shape_b, 2);
    
    Tensor* c = tensor_matmul(a, b);
    
    ASSERT_NULL(c);
    
    tensor_free(a);
    tensor_free(b);
}

void test_tensor_transpose() {
    TEST_HEADER("=== Test: tensor_transpose ===");
    
    int shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2);
    for (int i = 0; i < 6; i++) t->data[i] = (float)(i + 1);
    
    Tensor* transposed = tensor_transpose(t, 0, 1);
    
    ASSERT_NOT_NULL(transposed);
    ASSERT_EQ(transposed->shape[0], 3);
    ASSERT_EQ(transposed->shape[1], 2);
    
    ASSERT_EQ_FLOAT(tensor_get(transposed, (int[]){0, 0}), 1.0f, 1e-6f);
    ASSERT_EQ_FLOAT(tensor_get(transposed, (int[]){0, 1}), 4.0f, 1e-6f);
    ASSERT_EQ_FLOAT(tensor_get(transposed, (int[]){2, 1}), 6.0f, 1e-6f);
    
    tensor_free(t);
    tensor_free(transposed);
}

void test_tensor_det_2x2() {
    TEST_HEADER("=== Test: tensor_det_2x2 ===");
    
    int shape[2] = {2, 2};
    Tensor* t = tensor_create(shape, 2);
    
    t->data[0] = 1.0f; t->data[1] = 2.0f;
    t->data[2] = 3.0f; t->data[3] = 4.0f;
    
    float det = tensor_det(t);
    ASSERT_EQ_FLOAT(det, -2.0f, 1e-5f);
    
    tensor_free(t);
}

void test_tensor_det_identity() {
    TEST_HEADER("=== Test: tensor_det_identity ===");
    
    int shape[2] = {3, 3};
    Tensor* t = tensor_create(shape, 2);
    
    t->data[0] = 1.0f; t->data[1] = 0.0f; t->data[2] = 0.0f;
    t->data[3] = 0.0f; t->data[4] = 1.0f; t->data[5] = 0.0f;
    t->data[6] = 0.0f; t->data[7] = 0.0f; t->data[8] = 1.0f;
    
    float det = tensor_det(t);
    ASSERT_EQ_FLOAT(det, 1.0f, 1e-5f);
    
    tensor_free(t);
}

/* ========== MEMORY TESTS ========== */

void test_tensor_free_null() {
    TEST_HEADER("=== Test: tensor_free_null ===");
    
    tensor_free(NULL);
    ASSERT(1);
}

void test_tensor_clone_independence() {
    TEST_HEADER("=== Test: tensor_clone_independence ===");
    
    int shape[2] = {2, 2};
    Tensor* t = tensor_ones(shape, 2);
    Tensor* clone = tensor_clone(t);
    
    clone->data[0] = 999.0f;
    clone->shape[0] = 999;
    
    ASSERT_EQ_FLOAT(t->data[0], 1.0f, 1e-6f);
    ASSERT_EQ(t->shape[0], 2);
    
    tensor_free(t);
    tensor_free(clone);
}

/* ========== MAIN TEST RUNNER ========== */

int main() {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  MiniTorch v2 - Unit Test Suite        ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    
    printf("\n--- CREATION TESTS ---\n");
    test_tensor_create_1d();
    test_tensor_create_2d();
    test_tensor_create_3d();
    test_tensor_zeros();
    test_tensor_ones();
    test_tensor_clone();
    
    printf("\n--- INDEXING TESTS ---\n");
    test_tensor_get_index_1d();
    test_tensor_get_index_2d();
    test_tensor_get_set_values();
    
    printf("\n--- RESHAPE TESTS ---\n");
    test_tensor_reshape_1d_to_2d();
    test_tensor_reshape_2d_to_1d();
    test_tensor_reshape_size_mismatch();
    
    printf("\n--- ELEMENTWISE OPERATION TESTS ---\n");
    test_tensor_add_basic();
    test_tensor_add_shape_mismatch();
    test_tensor_add_broadcast_row_vector();
    test_tensor_sub();
    test_tensor_mul();
    test_tensor_div();
    test_tensor_sum_and_mean();
    
    printf("\n--- LINEAR ALGEBRA TESTS ---\n");
    test_tensor_matmul_basic();
    test_tensor_matmul_incompatible();
    test_tensor_transpose();
    test_tensor_det_2x2();
    test_tensor_det_identity();
    
    printf("\n--- MEMORY TESTS ---\n");
    test_tensor_free_null();
    test_tensor_clone_independence();
    
    PRINT_STATS();
    
    return TEST_RESULT();
}
