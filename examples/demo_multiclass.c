#include <stdio.h>
#include <stdlib.h>

#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

#define N_SAMPLES 9
#define N_FEATURES 2
#define N_CLASSES 3

static void init_model(AgTape* tape, MtLinear* model) {
    mt_linear_init_xavier_uniform(tape, model);
}

static float train_batch(AgTape* tape,
                         MtLinear* model,
                         MtOptimizer* optim,
                         int graph_checkpoint,
                         const MtBatch* batch,
                         int count) {
    ag_rewind(tape, graph_checkpoint);

    AgVal losses[N_SAMPLES];
    for (int i = 0; i < count; i++) {
        AgVal input[N_FEATURES] = {
            ag_leaf(tape, mt_batch_feature(batch, i, 0)),
            ag_leaf(tape, mt_batch_feature(batch, i, 1))
        };
        AgVal logits[N_CLASSES];
        AgVal target[N_CLASSES];
        int label = (int)mt_batch_label(batch, i);

        mt_linear_forward(tape, model, input, logits);
        for (int c = 0; c < N_CLASSES; c++) {
            target[c] = ag_leaf(tape, c == label ? 1.0f : 0.0f);
        }
        losses[i] = mt_cross_entropy_from_logits(tape, logits, target, N_CLASSES);
    }

    AgVal loss = ag_mul(tape, ag_sum(tape, losses, count), ag_leaf(tape, 1.0f / (float)count));
    float value = ag_data(tape, loss);

    mt_optimizer_zero_grad(tape, optim);
    ag_backward(tape, loss);
    mt_optimizer_step(tape, optim);
    return value;
}

static void predict_one(AgTape* tape, MtLinear* model, int graph_checkpoint, float x0, float x1, AgVal* probs) {
    ag_rewind(tape, graph_checkpoint);

    AgVal input[N_FEATURES] = {
        ag_leaf(tape, x0),
        ag_leaf(tape, x1)
    };
    AgVal logits[N_CLASSES];

    mt_linear_forward(tape, model, input, logits);
    mt_softmax(tape, logits, N_CLASSES, probs);
}

int main(void) {
    float x[] = {
        -2.0f, -2.0f,
        -2.0f, -1.0f,
        -1.0f, -2.0f,
         2.0f,  0.0f,
         2.0f,  1.0f,
         3.0f,  0.0f,
         0.0f,  2.0f,
         1.0f,  2.0f,
         0.0f,  3.0f
    };
    float y[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f
    };

    int epochs = 200;
    int batch_size = 3;
    float lr = 0.05f;

    printf("Démo classification multi-classe MiniTorch\n");
    printf("Modèle : Linear(2, 3) + CrossEntropyFromLogits\n");
    printf("\nÉpoques : ");
    scanf("%d", &epochs);
    if (epochs < 1) epochs = 200;

    printf("Taille de batch : ");
    scanf("%d", &batch_size);
    if (batch_size < 1) batch_size = 3;
    if (batch_size > N_SAMPLES) batch_size = N_SAMPLES;

    printf("Taux d'apprentissage : ");
    scanf("%f", &lr);
    if (lr <= 0.0f) lr = 0.05f;

    srand(11);

    AgTape* tape = ag_tape_create();
    MtDataset* dataset = mt_dataset_create(x, y, N_SAMPLES, N_FEATURES);
    MtBatch* batch = mt_batch_create(batch_size, N_FEATURES);
    MtLinear* model = mt_linear_create(tape, N_FEATURES, N_CLASSES, 1);
    MtOptimizer* optim = mt_adam_create(lr, 0.9f, 0.999f, 1e-8f);

    if (!tape || !dataset || !batch || !model || !optim) {
        printf("Impossible de créer la démo multi-classe.\n");
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

    int print_every = epochs / 10;
    if (print_every < 1) print_every = 1;

    printf("\nEntraînement\n");
    printf("Époque   Perte CE\n");
    printf("-----------------\n");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;
        int seen = 0;
        int n_batches = mt_dataset_num_batches(dataset, batch_size);

        mt_dataset_shuffle(dataset);
        for (int batch_idx = 0; batch_idx < n_batches; batch_idx++) {
            int count = mt_dataset_get_batch(dataset, batch_idx, batch_size, batch);
            if (count <= 0) continue;
            float loss = train_batch(tape, model, optim, graph_checkpoint, batch, count);
            total_loss += loss * (float)count;
            seen += count;
        }

        if (epoch == 1 || epoch % print_every == 0 || epoch == epochs) {
            printf("%-8d %.6f\n", epoch, total_loss / (float)seen);
        }
    }

    printf("\nPrédictions finales\n");
    printf("x0     x1     classe vraie   p0      p1      p2      classe\n");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < N_SAMPLES; i++) {
        AgVal probs[N_CLASSES];
        float x0 = mt_dataset_feature(dataset, i, 0);
        float x1 = mt_dataset_feature(dataset, i, 1);
        int label = (int)mt_dataset_label(dataset, i);
        predict_one(tape, model, graph_checkpoint, x0, x1, probs);
        int pred = mt_argmax(tape, probs, N_CLASSES);
        printf("%-6.1f %-6.1f %-13d %.3f   %.3f   %.3f   %d\n",
               x0, x1, label,
               ag_data(tape, probs[0]),
               ag_data(tape, probs[1]),
               ag_data(tape, probs[2]),
               pred);
    }

    int confusion[N_CLASSES * N_CLASSES];
    MtEvalResult eval;
    int ok_eval = mt_eval_multiclass_linear(tape, model, graph_checkpoint, dataset, N_CLASSES, &eval, confusion);

    if (ok_eval) {
        printf("Perte moyenne : %.6f\n", eval.loss_mean);
        printf("Exactitude : %.2f %% (%d/%d)\n", eval.accuracy * 100.0f, eval.correct, eval.total);
        printf("\nMatrice de confusion (lignes=vrai, colonnes=prédit)\n");
        for (int row = 0; row < N_CLASSES; row++) {
            for (int col = 0; col < N_CLASSES; col++) {
                printf("%4d", confusion[row * N_CLASSES + col]);
            }
            printf("\n");
        }
    }

    mt_optimizer_free(optim);
    mt_linear_free(model);
    mt_batch_free(batch);
    mt_dataset_free(dataset);
    ag_tape_free(tape);
    return 0;
}
