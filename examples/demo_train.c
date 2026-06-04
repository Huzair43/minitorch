#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minitorch/core/autograd.h"

#define MAX_SAMPLES 64
#define MAX_FEATURES 8

static float X[MAX_SAMPLES][MAX_FEATURES];
static float Y[MAX_SAMPLES];
static int n_samples = 0;
static int n_features = 0;

static float W[MAX_FEATURES];
static float B = 0.0f;

static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void init_model(void) {
    float scale = sqrtf(2.0f / (float)n_features);
    for (int i = 0; i < n_features; i++) {
        W[i] = randf(-scale, scale);
    }
    B = 0.0f;
}

static void input_dataset(void) {
    printf("\nPréparation du jeu de données\n");
    do {
        printf("Nombre de variables (1..%d) : ", MAX_FEATURES);
        scanf("%d", &n_features);
    } while (n_features < 1 || n_features > MAX_FEATURES);

    do {
        printf("Nombre d'exemples (2..%d) : ", MAX_SAMPLES);
        scanf("%d", &n_samples);
    } while (n_samples < 2 || n_samples > MAX_SAMPLES);

    printf("\nEntrez une étiquette binaire y dans [0, 1].\n");
    for (int i = 0; i < n_samples; i++) {
        printf("\nSample %d\n", i + 1);
        for (int f = 0; f < n_features; f++) {
            printf("  x%d = ", f + 1);
            scanf("%f", &X[i][f]);
        }
        float label = 0.0f;
        do {
            printf("  y  = ");
            scanf("%f", &label);
            if (label < 0.0f || label > 1.0f) {
                printf("  y doit être dans [0, 1]\n");
            }
        } while (label < 0.0f || label > 1.0f);
        Y[i] = label;
    }
}

static void print_dataset(void) {
    printf("\nJeu de données (%d exemples, %d variables)\n", n_samples, n_features);
    for (int i = 0; i < n_samples; i++) {
        printf("  exemple %d : [", i + 1);
        for (int f = 0; f < n_features; f++) {
            if (f > 0) printf(", ");
            printf("%.2f", X[i][f]);
        }
        printf("] -> y=%.1f\n", Y[i]);
    }
}

static float predict_raw(int idx) {
    float z = B;
    for (int f = 0; f < n_features; f++) {
        z += X[idx][f] * W[f];
    }
    return z;
}

static float predict_prob(int idx) {
    float z = predict_raw(idx);
    return 1.0f / (1.0f + expf(-z));
}

static float forward_backward_sample(int idx, float* dW, float* dB) {
    AgTape* tape = ag_tape_create();

    AgVal features[MAX_FEATURES];
    AgVal weights[MAX_FEATURES];
    for (int f = 0; f < n_features; f++) {
        features[f] = ag_leaf(tape, X[idx][f]);
        weights[f] = ag_leaf(tape, W[f]);
    }

    AgVal bias = ag_leaf(tape, B);
    AgVal z = bias;
    for (int f = 0; f < n_features; f++) {
        z = ag_add(tape, z, ag_mul(tape, features[f], weights[f]));
    }

    AgVal y_hat = ag_sigmoid(tape, z);
    AgVal target = ag_leaf(tape, Y[idx]);
    AgVal one = ag_leaf(tape, 1.0f);
    AgVal eps = ag_leaf(tape, 1e-7f);
    AgVal loss = ag_neg(tape,
        ag_add(tape,
            ag_mul(tape, target, ag_log(tape, ag_add(tape, y_hat, eps))),
            ag_mul(tape, ag_sub(tape, one, target),
                          ag_log(tape, ag_add(tape, ag_sub(tape, one, y_hat), eps)))
        )
    );

    float loss_value = ag_data(tape, loss);
    ag_backward(tape, loss);

    for (int f = 0; f < n_features; f++) {
        dW[f] = ag_grad(tape, weights[f]);
    }
    *dB = ag_grad(tape, bias);

    ag_tape_free(tape);
    return loss_value;
}

static void train(int epochs, float lr) {
    float dW[MAX_FEATURES];
    float dB = 0.0f;

    printf("\nEntraînement de la régression logistique\n");
    printf("Modèle : y_hat = sigmoid(sum(x_i * w_i) + b)\n");
    printf("Époque   Perte moy.\n");
    printf("-----------------\n");

    int print_every = epochs / 10;
    if (print_every < 1) print_every = 1;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;

        for (int i = 0; i < n_samples; i++) {
            float loss = forward_backward_sample(i, dW, &dB);
            total_loss += loss;

            for (int f = 0; f < n_features; f++) {
                W[f] -= lr * dW[f];
            }
            B -= lr * dB;
        }

        if (epoch == 1 || epoch % print_every == 0 || epoch == epochs) {
            printf("%-8d %.6f\n", epoch, total_loss / (float)n_samples);
        }
    }
}

static void print_results(float threshold) {
    printf("\nPrédictions finales (seuil = %.2f)\n", threshold);
    printf("exemple  y_vrai   proba    pred\n");
    printf("--------------------------------\n");

    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        float prob = predict_prob(i);
        int pred = prob >= threshold ? 1 : 0;
        int truth = Y[i] >= 0.5f ? 1 : 0;
        if (pred == truth) {
            correct++;
        }
        printf("%-8d %-8.1f %-8.4f %d\n", i + 1, Y[i], prob, pred);
    }

    printf("Précision : %d/%d\n", correct, n_samples);
}

int main(void) {
    printf("Démo d'entraînement MiniTorch\n");
    printf("Cette démo montre une régression logistique simple basée sur l'autograd.\n");

    input_dataset();
    print_dataset();

    int epochs = 0;
    float lr = 0.0f;
    float threshold = 0.5f;

    printf("\nHyperparamètres\n");
    printf("Époques : ");
    scanf("%d", &epochs);
    printf("Taux d'apprentissage : ");
    scanf("%f", &lr);
    printf("Seuil de classification (par défaut 0.5) : ");
    scanf("%f", &threshold);

    srand(42);
    init_model();

    printf("\nInitial weights\n");
    for (int f = 0; f < n_features; f++) {
        printf("  w%d = %.4f\n", f + 1, W[f]);
    }
    printf("  b  = %.4f\n", B);

    train(epochs, lr);

    printf("\nTrained weights\n");
    for (int f = 0; f < n_features; f++) {
        printf("  w%d = %.4f\n", f + 1, W[f]);
    }
    printf("  b  = %.4f\n", B);

    print_results(threshold);
    return 0;
}
