#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"
#include "minitorch/core/tensor_linalg.h"
#include "minitorch/core/autograd.h"

static void configure_console(void) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif
}

static void print_title(const char* title) {
    printf("\n=== %s ===\n", title);
}

static void print_tensor_2d(const Tensor* t) {
    if (!t || t->ndim != 2) {
        printf("<invalid tensor>\n");
        return;
    }

    for (int i = 0; i < t->shape[0]; i++) {
        printf("[");
        for (int j = 0; j < t->shape[1]; j++) {
            if (j > 0) printf(", ");
            printf("%7.3f", tensor_get(t, (int[]){i, j}));
        }
        printf(" ]\n");
    }
}

static void demo_tensor_basics(void) {
    print_title("Bases des tenseurs");

    int shape[2] = {2, 3};
    Tensor* a = tensor_create(shape, 2);
    Tensor* b = tensor_create(shape, 2);

    float a_values[] = {1, 2, 3, 4, 5, 6};
    float b_values[] = {10, 20, 30, 40, 50, 60};
    for (int i = 0; i < 6; i++) {
        a->data[i] = a_values[i];
        b->data[i] = b_values[i];
    }

    Tensor* sum = tensor_add(a, b);
    Tensor* prod = tensor_mul(a, b);

    printf("A:\n");
    print_tensor_2d(a);
    printf("B:\n");
    print_tensor_2d(b);
    printf("A + B:\n");
    print_tensor_2d(sum);
    printf("A * B:\n");
    print_tensor_2d(prod);

    tensor_free(a);
    tensor_free(b);
    tensor_free(sum);
    tensor_free(prod);
}

static void demo_broadcast_and_reduce(void) {
    print_title("Broadcast et réduction");

    int matrix_shape[2] = {2, 3};
    int bias_shape[1] = {3};
    Tensor* matrix = tensor_create(matrix_shape, 2);
    Tensor* bias = tensor_create(bias_shape, 1);

    float matrix_values[] = {1, 2, 3, 4, 5, 6};
    float bias_values[] = {10, 20, 30};
    for (int i = 0; i < 6; i++) matrix->data[i] = matrix_values[i];
    for (int i = 0; i < 3; i++) bias->data[i] = bias_values[i];

    Tensor* shifted = tensor_add(matrix, bias);
    Tensor* row_sum = tensor_sum(matrix, 1);
    Tensor* col_mean = tensor_mean(matrix, 0);
    Tensor* bias_broadcast = tensor_broadcast_to(bias, matrix_shape, 2);

    printf("Matrice :\n");
    print_tensor_2d(matrix);
    printf("Ligne de biais :\n");
    print_tensor_2d(bias_broadcast);
    printf("Matrice + biais :\n");
    print_tensor_2d(shifted);
    printf("Row sums (axis=1):\n");
    tensor_print(row_sum);
    printf("Column means (axis=0):\n");
    tensor_print(col_mean);

    tensor_free(matrix);
    tensor_free(bias);
    tensor_free(shifted);
    tensor_free(row_sum);
    tensor_free(col_mean);
    tensor_free(bias_broadcast);
}

static void demo_linalg(void) {
    print_title("Algèbre linéaire");

    int a_shape[2] = {2, 2};
    int b_shape[2] = {2, 2};
    Tensor* a = tensor_create(a_shape, 2);
    Tensor* b = tensor_create(b_shape, 2);

    float a_values[] = {1, 2, 3, 4};
    float b_values[] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) {
        a->data[i] = a_values[i];
        b->data[i] = b_values[i];
    }

    Tensor* c = tensor_matmul(a, b);
    Tensor* at = tensor_transpose(a, 0, 1);
    float det = tensor_det(a);

    printf("A:\n");
    print_tensor_2d(a);
    printf("B:\n");
    print_tensor_2d(b);
    printf("A @ B:\n");
    print_tensor_2d(c);
    printf("transpose(A):\n");
    print_tensor_2d(at);
    printf("det(A) = %.3f\n", det);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(at);
}

static void demo_scalar_autograd(void) {
    print_title("Chaîne autograd scalaire");

    AgTape* tape = ag_tape_create();
    AgVal x = ag_leaf(tape, 2.0f);
    AgVal w = ag_leaf(tape, 3.0f);
    AgVal b = ag_leaf(tape, 1.0f);
    AgVal y = ag_leaf(tape, 5.0f);

    AgVal y_hat = ag_add(tape, ag_mul(tape, x, w), b);
    AgVal diff = ag_sub(tape, y_hat, y);
    AgVal loss = ag_square(tape, diff);

    printf("Modèle : y_hat = x*w + b\n");
    printf("x=2, w=3, b=1, y=5\n");
    printf("y_hat = %.3f\n", ag_data(tape, y_hat));
    printf("perte = %.3f\n", ag_data(tape, loss));

    ag_backward(tape, loss);
    printf("dL/dx = %.3f\n", ag_grad(tape, x));
    printf("dL/dw = %.3f\n", ag_grad(tape, w));
    printf("dL/db = %.3f\n", ag_grad(tape, b));

    ag_tape_free(tape);
}

static void demo_tensor_autograd(void) {
    print_title("Aides autograd tensorielles");

    AgTape* tape = ag_tape_create();

    AgVal x[6];
    AgVal bias[3];
    AgVal shifted[6];

    float x_values[] = {1, 2, 3, 4, 5, 6};
    float bias_values[] = {10, 20, 30};
    for (int i = 0; i < 6; i++) x[i] = ag_leaf(tape, x_values[i]);
    for (int i = 0; i < 3; i++) bias[i] = ag_leaf(tape, bias_values[i]);

    ag_tensor_broadcast_binary_2d(tape, x, 2, 3, bias, 1, 3, shifted, ag_add);

    printf("X + bias:\n");
    for (int i = 0; i < 2; i++) {
        printf("[");
        for (int j = 0; j < 3; j++) {
            if (j > 0) printf(", ");
            printf("%7.3f", ag_data(tape, shifted[i * 3 + j]));
        }
        printf(" ]\n");
    }

    AgVal loss = ag_sum(tape, shifted, 6);
    printf("perte = sum(X + biais) = %.3f\n", ag_data(tape, loss));
    ag_backward(tape, loss);

    printf("Gradients pour le biais :\n");
    for (int j = 0; j < 3; j++) {
        printf("  db[%d] = %.3f\n", j, ag_grad(tape, bias[j]));
    }

    ag_tape_free(tape);
}

static void print_menu(void) {
    printf("\nMenu de démo MiniTorch\n");
    printf("1. Bases des tenseurs\n");
    printf("2. Broadcast et réduction\n");
    printf("3. Algèbre linéaire\n");
    printf("4. Chaîne autograd scalaire\n");
    printf("5. Aides autograd tensorielles\n");
    printf("0. Quit\n");
    printf("Choix : ");
}

int main(void) {
    configure_console();
    printf("Démo MiniTorch\n");

    while (1) {
        int choice = -1;
        print_menu();
        if (scanf("%d", &choice) != 1) {
            return 1;
        }

        switch (choice) {
            case 1: demo_tensor_basics(); break;
            case 2: demo_broadcast_and_reduce(); break;
            case 3: demo_linalg(); break;
            case 4: demo_scalar_autograd(); break;
            case 5: demo_tensor_autograd(); break;
            case 0:
                printf("Au revoir\n");
                return 0;
            default:
                printf("Choix invalide\n");
                break;
        }
    }
}
