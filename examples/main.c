#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"
#include "minitorch/core/tensor_linalg.h"
#include "minitorch/core/autograd.h"

/* ════════════════════════════════════════════════════════
   MENU PRINCIPAL
   ════════════════════════════════════════════════════════ */

void print_menu() {
    printf("\n=== MiniTorch v2 - Interactive Menu ===\n");
    printf("--- Tenseurs ---\n");
    printf("1. Creer et afficher une matrice\n");
    printf("2. Addition de deux matrices\n");
    printf("3. Multiplication element par element\n");
    printf("4. Multiplication matricielle (matmul)\n");
    printf("5. Transposer une matrice\n");
    printf("6. Calculer le determinant\n");
    printf("--- Autograd ---\n");
    printf("7. Demo : z = x * y, backward\n");
    printf("8. Demo : z = relu(x * w + b)\n");
    printf("9. Demo : regression (1 pas de gradient)\n");
    printf("0. Quitter\n");
    printf("Choisir une option: ");
}

/* ════════════════════════════════════════════════════════
   OPTIONS TENSEURS (inchangees)
   ════════════════════════════════════════════════════════ */

Tensor* input_matrix() {
    int rows, cols;
    printf("Nombre de lignes: ");
    scanf("%d", &rows);
    printf("Nombre de colonnes: ");
    scanf("%d", &cols);

    if (rows <= 0 || cols <= 0) {
        printf("Erreur: dimensions invalides!\n");
        return NULL;
    }

    int shape[2] = {rows, cols};
    Tensor* t = tensor_create(shape, 2);
    if (!t) {
        printf("Erreur: allocation memoire echouee\n");
        return NULL;
    }

    printf("Entrer les %d valeurs (ligne par ligne):\n", rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("  [%d][%d] = ", i, j);
            float value;
            scanf("%f", &value);
            tensor_set(t, (int[]){i, j}, value);
        }
    }
    return t;
}

void option_create_matrix() {
    printf("\n--- Creer une matrice ---\n");
    Tensor* t = input_matrix();
    if (t) {
        printf("\nMatrice creee:\n");
        tensor_print(t);
        tensor_free(t);
    }
}

void option_add_matrices() {
    printf("\n--- Addition de deux matrices ---\n");
    printf("\nPremiere matrice:\n");
    Tensor* a = input_matrix();
    if (!a) return;
    printf("\nDeuxieme matrice:\n");
    Tensor* b = input_matrix();
    if (!b) { tensor_free(a); return; }
    Tensor* c = tensor_add(a, b);
    if (c) {
        printf("\nMatrice A:\n"); tensor_print(a);
        printf("\nMatrice B:\n"); tensor_print(b);
        printf("\nA + B:\n");     tensor_print(c);
        tensor_free(c);
    } else {
        printf("Erreur: les matrices doivent avoir les memes dimensions!\n");
    }
    tensor_free(a);
    tensor_free(b);
}

void option_mul_elementwise() {
    printf("\n--- Multiplication element par element ---\n");
    printf("\nPremiere matrice:\n");
    Tensor* a = input_matrix();
    if (!a) return;
    printf("\nDeuxieme matrice:\n");
    Tensor* b = input_matrix();
    if (!b) { tensor_free(a); return; }
    Tensor* c = tensor_mul(a, b);
    if (c) {
        printf("\nMatrice A:\n"); tensor_print(a);
        printf("\nMatrice B:\n"); tensor_print(b);
        printf("\nA * B (element par element):\n"); tensor_print(c);
        tensor_free(c);
    } else {
        printf("Erreur: les matrices doivent avoir les memes dimensions!\n");
    }
    tensor_free(a);
    tensor_free(b);
}

void option_matmul() {
    printf("\n--- Multiplication matricielle ---\n");
    printf("\nMatrice A (m x k):\n");
    Tensor* a = input_matrix();
    if (!a) return;
    printf("\nMatrice B (k x n):\n");
    Tensor* b = input_matrix();
    if (!b) { tensor_free(a); return; }
    Tensor* c = tensor_matmul(a, b);
    if (c) {
        printf("\nMatrice A (%d x %d):\n", a->shape[0], a->shape[1]); tensor_print(a);
        printf("\nMatrice B (%d x %d):\n", b->shape[0], b->shape[1]); tensor_print(b);
        printf("\nA @ B (%d x %d):\n",     c->shape[0], c->shape[1]); tensor_print(c);
        tensor_free(c);
    }
    tensor_free(a);
    tensor_free(b);
}

void option_transpose() {
    printf("\n--- Transposer une matrice ---\n");
    Tensor* t = input_matrix();
    if (!t) return;
    if (t->ndim != 2) {
        printf("Erreur: transpose fonctionne uniquement sur les matrices 2D\n");
        tensor_free(t); return;
    }
    Tensor* transposed = tensor_transpose(t, 0, 1);
    if (transposed) {
        printf("\nMatrice originale (%d x %d):\n",   t->shape[0], t->shape[1]);         tensor_print(t);
        printf("\nMatrice transposee (%d x %d):\n", transposed->shape[0], transposed->shape[1]); tensor_print(transposed);
        tensor_free(transposed);
    }
    tensor_free(t);
}

void option_determinant() {
    printf("\n--- Calculer le determinant ---\n");
    Tensor* t = input_matrix();
    if (!t) return;
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) {
        printf("Erreur: le determinant s'applique uniquement aux matrices carrees!\n");
        tensor_free(t); return;
    }
    float det = tensor_det(t);
    printf("\nMatrice %d x %d:\n", t->shape[0], t->shape[1]);
    tensor_print(t);
    printf("\nDeterminant = %f\n", det);
    tensor_free(t);
}

/* ════════════════════════════════════════════════════════
   OPTIONS AUTOGRAD
   ════════════════════════════════════════════════════════ */

/*
 * Option 7 : z = x * y
 * Verifie que dz/dx = y et dz/dy = x
 */
void option_ag_mul_simple() {
    printf("\n--- Autograd : z = x * y ---\n");

    float xv, yv;
    printf("x = "); scanf("%f", &xv);
    printf("y = "); scanf("%f", &yv);

    AgTape *t = ag_tape_create();

    AgVal x = ag_leaf(t, xv);
    AgVal y = ag_leaf(t, yv);
    AgVal z = ag_mul(t, x, y);

    printf("\nForward:\n");
    printf("  x       = %.4f\n", ag_data(t, x));
    printf("  y       = %.4f\n", ag_data(t, y));
    printf("  z = x*y = %.4f\n", ag_data(t, z));

    ag_backward(t, z);

    printf("\nBackward:\n");
    printf("  dz/dx = %.4f  (attendu : y = %.4f)\n", ag_grad(t, x), yv);
    printf("  dz/dy = %.4f  (attendu : x = %.4f)\n", ag_grad(t, y), xv);

    ag_tape_free(t);
}

/*
 * Option 8 : z = relu(x * w + b)
 * Neurone unique, activation relu
 */
void option_ag_neuron() {
    printf("\n--- Autograd : z = relu(x*w + b) ---\n");

    float xv, wv, bv;
    printf("x = "); scanf("%f", &xv);
    printf("w = "); scanf("%f", &wv);
    printf("b = "); scanf("%f", &bv);

    AgTape *t = ag_tape_create();

    AgVal x   = ag_leaf(t, xv);
    AgVal w   = ag_leaf(t, wv);
    AgVal b   = ag_leaf(t, bv);
    AgVal xw  = ag_mul(t, x, w);
    AgVal pre = ag_add(t, xw, b);
    AgVal z   = ag_relu(t, pre);

    printf("\nForward:\n");
    printf("  x*w         = %.4f\n", ag_data(t, xw));
    printf("  x*w + b     = %.4f\n", ag_data(t, pre));
    printf("  relu(x*w+b) = %.4f\n", ag_data(t, z));

    ag_backward(t, z);

    printf("\nBackward (gradients par rapport a z):\n");
    printf("  dz/dw = %.4f\n", ag_grad(t, w));
    printf("  dz/db = %.4f\n", ag_grad(t, b));
    printf("  dz/dx = %.4f\n", ag_grad(t, x));

    if (ag_data(t, pre) > 0.0f) {
        printf("\n  [relu actif]  dz/dw attendu = x = %.4f\n", xv);
        printf("                dz/db attendu = 1\n");
    } else {
        printf("\n  [relu eteint] tous les gradients attendus = 0\n");
    }

    ag_tape_free(t);
}

/*
 * Option 9 : regression scalaire — 1 pas de gradient
 *
 * Modele  : y_hat = x * w + b
 * Loss    : L = (y_hat - y_true)^2
 * Mise a jour : w <- w - lr * dL/dw
 *               b <- b - lr * dL/db
 */
void option_ag_regression_step() {
    printf("\n--- Autograd : regression, 1 pas ---\n");
    printf("Modele : y_hat = x*w + b,  Loss = (y_hat - y_true)^2\n\n");

    float xv, wv, bv, ytrue, lr;
    printf("x      = "); scanf("%f", &xv);
    printf("w      = "); scanf("%f", &wv);
    printf("b      = "); scanf("%f", &bv);
    printf("y_true = "); scanf("%f", &ytrue);
    printf("lr     = "); scanf("%f", &lr);

    AgTape *t = ag_tape_create();

    AgVal x      = ag_leaf(t, xv);
    AgVal w      = ag_leaf(t, wv);
    AgVal b      = ag_leaf(t, bv);
    AgVal y_true = ag_leaf(t, ytrue);

    AgVal y_hat  = ag_add(t, ag_mul(t, x, w), b);
    AgVal diff   = ag_sub(t, y_hat, y_true);
    AgVal loss   = ag_mul(t, diff, diff);   /* (y_hat - y_true)^2 */

    printf("\nForward:\n");
    printf("  y_hat = %.4f\n", ag_data(t, y_hat));
    printf("  loss  = %.4f\n", ag_data(t, loss));

    ag_backward(t, loss);

    float dw = ag_grad(t, w);
    float db = ag_grad(t, b);

    printf("\nBackward:\n");
    printf("  dL/dw = %.4f\n", dw);
    printf("  dL/db = %.4f\n", db);

    float new_w = wv - lr * dw;
    float new_b = bv - lr * db;

    printf("\nApres mise a jour (lr = %.4f):\n", lr);
    printf("  w : %.4f  ->  %.4f\n", wv, new_w);
    printf("  b : %.4f  ->  %.4f\n", bv, new_b);

    /* Verification : loss apres le pas */
    float new_yhat = xv * new_w + new_b;
    float new_loss = (new_yhat - ytrue) * (new_yhat - ytrue);
    printf("\nVerification (calcul direct):\n");
    printf("  new y_hat = %.4f\n", new_yhat);
    printf("  new loss  = %.4f  (avant : %.4f)\n", new_loss, ag_data(t, loss));

    ag_tape_free(t);
}

/* ════════════════════════════════════════════════════════
   MAIN
   ════════════════════════════════════════════════════════ */

int main() {
    SetConsoleOutputCP(65001);
    printf("╔════════════════════════════════════════╗\n");
    printf("║    MiniTorch v2 - Interactive Mode    ║\n");
    printf("╚════════════════════════════════════════╝\n");

    int choice;
    while (1) {
        print_menu();
        scanf("%d", &choice);

        switch (choice) {
            case 1: option_create_matrix();     break;
            case 2: option_add_matrices();      break;
            case 3: option_mul_elementwise();   break;
            case 4: option_matmul();            break;
            case 5: option_transpose();         break;
            case 6: option_determinant();       break;
            case 7: option_ag_mul_simple();     break;
            case 8: option_ag_neuron();         break;
            case 9: option_ag_regression_step(); break;
            case 0:
                printf("\nMerci d'avoir utilise MiniTorch! Au revoir.\n");
                return 0;
            default:
                printf("Option invalide. Veuillez reessayer.\n");
        }
    }
    return 0;
}
