#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

#define MAX_SAMPLES 64
#define MAX_FEATURES 8

static float X[MAX_SAMPLES][MAX_FEATURES];
static float Y[MAX_SAMPLES];
static int n_samples = 0;
static int n_features = 0;

static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void init_model(AgTape* tape, MtLinear* model) {
    float scale = sqrtf(2.0f / (float)n_features);
    for (int i = 0; i < n_features; i++) {
        mt_linear_set_weight(tape, model, 0, i, randf(-scale, scale));
    }
    mt_linear_set_bias(tape, model, 0, 0.0f);
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
        printf("\nExemple %d\n", i + 1);
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

static float predict_raw(AgTape* tape, const MtLinear* model, int idx) {
    float z = ag_data(tape, mt_linear_bias(model, 0));
    for (int f = 0; f < n_features; f++) {
        z += X[idx][f] * ag_data(tape, mt_linear_weight(model, 0, f));
    }
    return z;
}

static float predict_prob(AgTape* tape, const MtLinear* model, int idx) {
    float z = predict_raw(tape, model, idx);
    return 1.0f / (1.0f + expf(-z));
}

static float train_sample(AgTape* tape, MtLinear* model, MtOptimizer* optim, int idx) {
    AgVal features[MAX_FEATURES];
    for (int f = 0; f < n_features; f++) {
        features[f] = ag_leaf(tape, X[idx][f]);
    }

    AgVal logits[1];
    AgVal pred[1];
    AgVal target[1] = {ag_leaf(tape, Y[idx])};

    mt_linear_forward(tape, model, features, logits);
    mt_sigmoid(tape, logits, 1, pred);
    AgVal loss = mt_bce_loss(tape, pred, target, 1);

    float loss_value = ag_data(tape, loss);
    mt_optimizer_zero_grad(tape, optim);
    ag_backward(tape, loss);
    mt_optimizer_step(tape, optim);
    return loss_value;
}

static void train(AgTape* tape, MtLinear* model, MtOptimizer* optim, int epochs) {
    printf("\nEntraînement de la régression logistique\n");
    printf("Modèle : Linear(%d, 1) + Sigmoid + BCE + Adam\n", n_features);
    printf("Époque   Perte moy.\n");
    printf("-----------------\n");

    int print_every = epochs / 10;
    if (print_every < 1) print_every = 1;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;

        for (int i = 0; i < n_samples; i++) {
            float loss = train_sample(tape, model, optim, i);
            total_loss += loss;
        }

        if (epoch == 1 || epoch % print_every == 0 || epoch == epochs) {
            printf("%-8d %.6f\n", epoch, total_loss / (float)n_samples);
        }
    }
}

static void print_results(AgTape* tape, const MtLinear* model, float threshold) {
    printf("\nPrédictions finales (seuil = %.2f)\n", threshold);
    printf("exemple  y_vrai   proba    classe\n");
    printf("--------------------------------\n");

    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        float prob = predict_prob(tape, model, i);
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
    AgTape* tape = ag_tape_create();
    MtLinear* model = mt_linear_create(tape, n_features, 1, 1);
    MtOptimizer* optim = mt_adam_create(lr, 0.9f, 0.999f, 1e-8f);
    if (!tape || !model || !optim) {
        printf("Impossible de créer le modèle nn.\n");
        mt_optimizer_free(optim);
        mt_linear_free(model);
        ag_tape_free(tape);
        return 1;
    }
    init_model(tape, model);
    mt_optimizer_add_linear(optim, model);

    printf("\nPoids initiaux\n");
    for (int f = 0; f < n_features; f++) {
        printf("  w%d = %.4f\n", f + 1, ag_data(tape, mt_linear_weight(model, 0, f)));
    }
    printf("  b  = %.4f\n", ag_data(tape, mt_linear_bias(model, 0)));

    train(tape, model, optim, epochs);

    printf("\nPoids entraînés\n");
    for (int f = 0; f < n_features; f++) {
        printf("  w%d = %.4f\n", f + 1, ag_data(tape, mt_linear_weight(model, 0, f)));
    }
    printf("  b  = %.4f\n", ag_data(tape, mt_linear_bias(model, 0)));

    print_results(tape, model, threshold);
    mt_optimizer_free(optim);
    mt_linear_free(model);
    ag_tape_free(tape);
    return 0;
}
