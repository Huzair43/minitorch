#include <stdio.h>
#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

#define XOR_SAMPLES 4
#define XOR_FEATURES 2
#define XOR_HIDDEN 4

static void init_xor_model(AgTape* tape, MtLinear* l1, MtLinear* l2) {
    mt_linear_set_weight(tape, l1, 0, 0, 3.0f);
    mt_linear_set_weight(tape, l1, 0, 1, 3.0f);
    mt_linear_set_bias(tape, l1, 0, -1.5f);

    mt_linear_set_weight(tape, l1, 1, 0, 3.0f);
    mt_linear_set_weight(tape, l1, 1, 1, 3.0f);
    mt_linear_set_bias(tape, l1, 1, -4.5f);

    for (int out_idx = 2; out_idx < XOR_HIDDEN; out_idx++) {
        mt_linear_set_weight(tape, l1, out_idx, 0, 0.0f);
        mt_linear_set_weight(tape, l1, out_idx, 1, 0.0f);
        mt_linear_set_bias(tape, l1, out_idx, 0.0f);
    }

    mt_linear_set_weight(tape, l2, 0, 0, 4.0f);
    mt_linear_set_weight(tape, l2, 0, 1, -4.0f);
    mt_linear_set_weight(tape, l2, 0, 2, 0.0f);
    mt_linear_set_weight(tape, l2, 0, 3, 0.0f);
    mt_linear_set_bias(tape, l2, 0, -1.5f);
}

static float forward_prob(AgTape* tape, const MtSequential* model, float x0, float x1) {
    AgVal input[2] = {
        ag_leaf(tape, x0),
        ag_leaf(tape, x1)
    };
    AgVal output[1];

    if (!mt_sequential_forward(tape, model, input, 2, output, 1)) {
        return 0.0f;
    }
    return ag_data(tape, output[0]);
}

static float train_batch(AgTape* tape,
                         MtSequential* model,
                         MtOptimizer* optim,
                         const MtBatch* batch,
                         int count) {
    AgVal pred[XOR_SAMPLES];
    AgVal target[XOR_SAMPLES];

    for (int i = 0; i < count; i++) {
        AgVal input[2] = {
            ag_leaf(tape, mt_batch_feature(batch, i, 0)),
            ag_leaf(tape, mt_batch_feature(batch, i, 1))
        };
        AgVal output[1];

        mt_sequential_forward(tape, model, input, 2, output, 1);
        pred[i] = output[0];
        target[i] = ag_leaf(tape, mt_batch_label(batch, i));
    }

    AgVal loss = mt_bce_loss(tape, pred, target, count);
    float value = ag_data(tape, loss);

    mt_optimizer_zero_grad(tape, optim);
    ag_backward(tape, loss);
    mt_optimizer_step(tape, optim);

    return value;
}

static void print_predictions(AgTape* tape, const MtSequential* model, const MtDataset* dataset) {
    printf("\nPrédictions XOR\n");
    printf("x0  x1  y_vrai  proba   classe\n");
    printf("-------------------------------\n");

    int correct = 0;
    for (int i = 0; i < dataset->n_samples; i++) {
        float x0 = mt_dataset_feature(dataset, i, 0);
        float x1 = mt_dataset_feature(dataset, i, 1);
        float label = mt_dataset_label(dataset, i);
        float prob = forward_prob(tape, model, x0, x1);
        int pred = prob >= 0.5f ? 1 : 0;
        int truth = label >= 0.5f ? 1 : 0;
        if (pred == truth) {
            correct++;
        }
        printf("%.0f   %.0f   %.0f       %.4f  %d\n", x0, x1, label, prob, pred);
    }

    printf("Précision : %d/%d\n", correct, dataset->n_samples);
}

int main(void) {
    float x[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    float y[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    int epochs = 100;
    float lr = 0.05f;

    printf("Démo MLP XOR MiniTorch\n");
    printf("Réseau : Linear(2, 4) + Tanh + Linear(4, 1) + Sigmoid\n");
    printf("Époques par défaut : %d\n", epochs);
    printf("Taux d'apprentissage par défaut : %.3f\n", lr);

    printf("\nÉpoques : ");
    scanf("%d", &epochs);
    if (epochs < 1) epochs = 100;

    printf("Taux d'apprentissage : ");
    scanf("%f", &lr);
    if (lr <= 0.0f) lr = 0.05f;

    AgTape* tape = ag_tape_create();
    MtDataset* dataset = mt_dataset_create(x, y, XOR_SAMPLES, XOR_FEATURES);
    MtBatch* batch = mt_batch_create(XOR_SAMPLES, XOR_FEATURES);
    MtLinear* l1 = mt_linear_create(tape, XOR_FEATURES, XOR_HIDDEN, 1);
    MtLinear* l2 = mt_linear_create(tape, XOR_HIDDEN, 1, 1);
    MtSequential* model = mt_sequential_create(XOR_HIDDEN);
    MtOptimizer* optim = mt_adam_create(lr, 0.9f, 0.999f, 1e-8f);

    if (!tape || !dataset || !batch || !l1 || !l2 || !model || !optim) {
        printf("Impossible de créer la démo XOR.\n");
        mt_optimizer_free(optim);
        mt_sequential_free(model);
        mt_linear_free(l2);
        mt_linear_free(l1);
        mt_batch_free(batch);
        mt_dataset_free(dataset);
        ag_tape_free(tape);
        return 1;
    }

    init_xor_model(tape, l1, l2);
    mt_sequential_add_linear(model, l1);
    mt_sequential_add_activation(model, MT_ACT_TANH);
    mt_sequential_add_linear(model, l2);
    mt_sequential_add_activation(model, MT_ACT_SIGMOID);
    mt_optimizer_add_sequential(optim, model);

    int print_every = epochs / 10;
    if (print_every < 1) print_every = 1;

    printf("\nEntraînement\n");
    printf("Époque   Perte BCE\n");
    printf("------------------\n");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        mt_dataset_shuffle(dataset);
        int count = mt_dataset_get_batch(dataset, 0, XOR_SAMPLES, batch);
        float loss = train_batch(tape, model, optim, batch, count);

        if (epoch == 1 || epoch % print_every == 0 || epoch == epochs) {
            printf("%-8d %.6f\n", epoch, loss);
        }
    }

    print_predictions(tape, model, dataset);

    mt_optimizer_free(optim);
    mt_sequential_free(model);
    mt_linear_free(l2);
    mt_linear_free(l1);
    mt_batch_free(batch);
    mt_dataset_free(dataset);
    ag_tape_free(tape);
    return 0;
}
