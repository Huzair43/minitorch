#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

#define MAX_SAMPLES 64
#define MAX_FEATURES 8

static float X[MAX_SAMPLES * MAX_FEATURES];
static float Y[MAX_SAMPLES];
static int n_samples = 0;
static int n_features = 0;

static void init_model(AgTape* tape, MtLinear* model) {
    mt_linear_init_xavier_uniform(tape, model);
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
            scanf("%f", &X[i * n_features + f]);
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
            printf("%.2f", X[i * n_features + f]);
        }
        printf("] -> y=%.1f\n", Y[i]);
    }
}

static float predict_raw(AgTape* tape, const MtLinear* model, const MtDataset* dataset, int idx) {
    float z = ag_data(tape, mt_linear_bias(model, 0));
    for (int f = 0; f < n_features; f++) {
        z += mt_dataset_feature(dataset, idx, f) * ag_data(tape, mt_linear_weight(model, 0, f));
    }
    return z;
}

static float predict_prob(AgTape* tape, const MtLinear* model, const MtDataset* dataset, int idx) {
    float z = predict_raw(tape, model, dataset, idx);
    return 1.0f / (1.0f + expf(-z));
}

static float train_batch(AgTape* tape, MtLinear* model, MtOptimizer* optim, int graph_checkpoint, const MtBatch* batch, int count) {
    ag_rewind(tape, graph_checkpoint);

    AgVal pred[MAX_SAMPLES];
    AgVal target[MAX_SAMPLES];
    AgVal features[MAX_FEATURES];
    AgVal logits[1];

    for (int i = 0; i < count; i++) {
        for (int f = 0; f < n_features; f++) {
            features[f] = ag_leaf(tape, mt_batch_feature(batch, i, f));
        }

        AgVal prob[1];
        mt_linear_forward(tape, model, features, logits);
        mt_sigmoid(tape, logits, 1, prob);
        pred[i] = prob[0];
        target[i] = ag_leaf(tape, mt_batch_label(batch, i));
    }

    AgVal loss = mt_bce_loss(tape, pred, target, count);

    float loss_value = ag_data(tape, loss);
    mt_optimizer_zero_grad(tape, optim);
    ag_backward(tape, loss);
    mt_optimizer_step(tape, optim);
    return loss_value;
}

static void train(AgTape* tape, MtLinear* model, MtOptimizer* optim, int graph_checkpoint, MtDataset* dataset, MtBatch* batch, int batch_size, int epochs) {
    printf("\nEntraînement de la régression logistique\n");
    printf("Modèle : Linear(%d, 1) + Sigmoid + BCE + Adam\n", n_features);
    printf("Batch : %d exemple(s)\n", batch_size);
    printf("Époque   Perte moy.\n");
    printf("-----------------\n");

    int print_every = epochs / 10;
    if (print_every < 1) print_every = 1;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;
        int seen = 0;
        int n_batches = mt_dataset_num_batches(dataset, batch_size);

        mt_dataset_shuffle(dataset);
        for (int batch_idx = 0; batch_idx < n_batches; batch_idx++) {
            int count = mt_dataset_get_batch(dataset, batch_idx, batch_size, batch);
            if (count <= 0) {
                continue;
            }
            float loss = train_batch(tape, model, optim, graph_checkpoint, batch, count);
            total_loss += loss * (float)count;
            seen += count;
        }

        if (epoch == 1 || epoch % print_every == 0 || epoch == epochs) {
            printf("%-8d %.6f\n", epoch, total_loss / (float)seen);
        }
    }
}

static void print_results(AgTape* tape, const MtLinear* model, int graph_checkpoint, const MtDataset* dataset, float threshold) {
    printf("\nPrédictions finales (seuil = %.2f)\n", threshold);
    printf("exemple  y_vrai   proba    classe\n");
    printf("--------------------------------\n");

    for (int i = 0; i < n_samples; i++) {
        float prob = predict_prob(tape, model, dataset, i);
        int pred = prob >= threshold ? 1 : 0;
        float label = mt_dataset_label(dataset, i);
        printf("%-8d %-8.1f %-8.4f %d\n", i + 1, label, prob, pred);
    }

    MtEvalResult eval;
    if (mt_eval_binary_linear(tape, model, graph_checkpoint, dataset, threshold, &eval)) {
        printf("Perte moyenne : %.6f\n", eval.loss_mean);
        printf("Exactitude : %.2f %% (%d/%d)\n", eval.accuracy * 100.0f, eval.correct, eval.total);
    }
}

int main(void) {
    printf("Démo d'entraînement MiniTorch\n");
    printf("Cette démo montre une régression logistique simple basée sur l'autograd.\n");

    input_dataset();
    print_dataset();

    int epochs = 0;
    int batch_size = 1;
    float lr = 0.0f;
    float threshold = 0.5f;

    printf("\nHyperparamètres\n");
    printf("Époques : ");
    scanf("%d", &epochs);
    printf("Taille de batch : ");
    scanf("%d", &batch_size);
    if (batch_size < 1) batch_size = 1;
    if (batch_size > n_samples) batch_size = n_samples;
    printf("Taux d'apprentissage : ");
    scanf("%f", &lr);
    printf("Seuil de classification (par défaut 0.5) : ");
    scanf("%f", &threshold);

    srand(42);
    AgTape* tape = ag_tape_create();
    MtDataset* dataset = mt_dataset_create(X, Y, n_samples, n_features);
    MtBatch* batch = mt_batch_create(batch_size, n_features);
    MtLinear* model = mt_linear_create(tape, n_features, 1, 1);
    MtOptimizer* optim = mt_adam_create(lr, 0.9f, 0.999f, 1e-8f);
    if (!tape || !dataset || !batch || !model || !optim) {
        printf("Impossible de créer le modèle nn.\n");
        mt_optimizer_free(optim);
        mt_linear_free(model);
        mt_batch_free(batch);
        mt_dataset_free(dataset);
        ag_tape_free(tape);
        return 1;
    }
    init_model(tape, model);
    mt_optimizer_add_linear(optim, model);
    int graph_checkpoint = ag_checkpoint(tape);

    printf("\nPoids initiaux\n");
    for (int f = 0; f < n_features; f++) {
        printf("  w%d = %.4f\n", f + 1, ag_data(tape, mt_linear_weight(model, 0, f)));
    }
    printf("  b  = %.4f\n", ag_data(tape, mt_linear_bias(model, 0)));

    train(tape, model, optim, graph_checkpoint, dataset, batch, batch_size, epochs);

    printf("\nPoids entraînés\n");
    for (int f = 0; f < n_features; f++) {
        printf("  w%d = %.4f\n", f + 1, ag_data(tape, mt_linear_weight(model, 0, f)));
    }
    printf("  b  = %.4f\n", ag_data(tape, mt_linear_bias(model, 0)));

    print_results(tape, model, graph_checkpoint, dataset, threshold);
    mt_optimizer_free(optim);
    mt_linear_free(model);
    mt_batch_free(batch);
    mt_dataset_free(dataset);
    ag_tape_free(tape);
    return 0;
}
