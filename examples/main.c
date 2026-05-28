#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"
#include "minitorch/core/tensor_linalg.h"

void print_menu() {
    printf("\n=== MiniTorch v2 - Interactive Menu ===\n");
    printf("1. Créer et afficher une matrice\n");
    printf("2. Addition de deux matrices\n");
    printf("3. Multiplication élément par élément\n");
    printf("4. Multiplication matricielle (matmul)\n");
    printf("5. Transposer une matrice\n");
    printf("6. Calculer le déterminant\n");
    printf("0. Quitter\n");
    printf("Choisir une option: ");
}

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
        printf("Erreur: allocation mémoire échouée\n");
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
    printf("\n--- Créer une matrice ---\n");
    Tensor* t = input_matrix();
    if (t) {
        printf("\nMatrice créée:\n");
        tensor_print(t);
        tensor_free(t);
    }
}

void option_add_matrices() {
    printf("\n--- Addition de deux matrices ---\n");
    
    printf("\nPremière matrice:\n");
    Tensor* a = input_matrix();
    if (!a) return;
    
    printf("\nDeuxième matrice:\n");
    Tensor* b = input_matrix();
    if (!b) {
        tensor_free(a);
        return;
    }
    
    Tensor* c = tensor_add(a, b);
    if (c) {
        printf("\nMatrice A:\n");
        tensor_print(a);
        printf("\nMatrice B:\n");
        tensor_print(b);
        printf("\nA + B:\n");
        tensor_print(c);
        tensor_free(c);
    } else {
        printf("Erreur: les matrices doivent avoir les mêmes dimensions!\n");
    }
    
    tensor_free(a);
    tensor_free(b);
}

void option_mul_elementwise() {
    printf("\n--- Multiplication élément par élément ---\n");
    
    printf("\nPremière matrice:\n");
    Tensor* a = input_matrix();
    if (!a) return;
    
    printf("\nDeuxième matrice:\n");
    Tensor* b = input_matrix();
    if (!b) {
        tensor_free(a);
        return;
    }
    
    Tensor* c = tensor_mul(a, b);
    if (c) {
        printf("\nMatrice A:\n");
        tensor_print(a);
        printf("\nMatrice B:\n");
        tensor_print(b);
        printf("\nA * B (élément par élément):\n");
        tensor_print(c);
        tensor_free(c);
    } else {
        printf("Erreur: les matrices doivent avoir les mêmes dimensions!\n");
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
    if (!b) {
        tensor_free(a);
        return;
    }
    
    Tensor* c = tensor_matmul(a, b);
    if (c) {
        printf("\nMatrice A (%d x %d):\n", a->shape[0], a->shape[1]);
        tensor_print(a);
        printf("\nMatrice B (%d x %d):\n", b->shape[0], b->shape[1]);
        tensor_print(b);
        printf("\nA @ B (%d x %d):\n", c->shape[0], c->shape[1]);
        tensor_print(c);
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
        tensor_free(t);
        return;
    }
    
    Tensor* transposed = tensor_transpose(t, 0, 1);
    if (transposed) {
        printf("\nMatrice originale (%d x %d):\n", t->shape[0], t->shape[1]);
        tensor_print(t);
        printf("\nMatrice transposée (%d x %d):\n", transposed->shape[0], transposed->shape[1]);
        tensor_print(transposed);
        tensor_free(transposed);
    }
    
    tensor_free(t);
}

void option_determinant() {
    printf("\n--- Calculer le déterminant ---\n");
    
    Tensor* t = input_matrix();
    if (!t) return;
    
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) {
        printf("Erreur: le déterminant s'applique uniquement aux matrices carrées!\n");
        tensor_free(t);
        return;
    }
    
    float det = tensor_det(t);
    printf("\nMatrice %d x %d:\n", t->shape[0], t->shape[1]);
    tensor_print(t);
    printf("\nDéterminant = %f\n", det);
    
    tensor_free(t);
}

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
            case 1:
                option_create_matrix();
                break;
            case 2:
                option_add_matrices();
                break;
            case 3:
                option_mul_elementwise();
                break;
            case 4:
                option_matmul();
                break;
            case 5:
                option_transpose();
                break;
            case 6:
                option_determinant();
                break;
            case 0:
                printf("\nMerci d'avoir utilisé MiniTorch! Au revoir.\n");
                return 0;
            default:
                printf("Option invalide. Veuillez réessayer.\n");
        }
    }
    
    return 0;
}
