#include <stdio.h>
#include <math.h>

#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

#define XOR_SAMPLES 4
#define XOR_FEATURES 2
#define XOR_HIDDEN 4

static int _passed = 0;
static int _failed = 0;

static void _check(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff <= tol) {
        printf("  [PASS] %-40s got=%.6f\n", name, got);
        _passed++;
    } else {
        printf("  [FAIL] %-40s got=%.6f  expected=%.6f  diff=%.2e\n",
               name, got, expected, diff);
        _failed++;
    }
}

#define CHECK(name, got, expected) _check(name, got, expected, 1e-4f)

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

static float eval_prob(AgTape* tape, const MtSequential* model, const MtDataset* dataset, int idx) {
    AgVal input[2] = {
        ag_leaf(tape, mt_dataset_feature(dataset, idx, 0)),
        ag_leaf(tape, mt_dataset_feature(dataset, idx, 1))
    };
    AgVal output[1];
    mt_sequential_forward(tape, model, input, 2, output, 1);
    return ag_data(tape, output[0]);
}

static float train_epoch(AgTape* tape,
                         MtSequential* model,
                         MtOptimizer* optim,
                         MtDataset* dataset,
                         MtBatch* batch) {
    AgVal pred[XOR_SAMPLES];
    AgVal target[XOR_SAMPLES];

    mt_dataset_shuffle(dataset);
    int count = mt_dataset_get_batch(dataset, 0, XOR_SAMPLES, batch);

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

static void test_mlp_learns_xor(void) {
    printf("\n-- MLP : apprentissage XOR\n");

    float x[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    float y[] = {0.0f, 1.0f, 1.0f, 0.0f};

    AgTape* tape = ag_tape_create();
    MtDataset* dataset = mt_dataset_create(x, y, XOR_SAMPLES, XOR_FEATURES);
    MtBatch* batch = mt_batch_create(XOR_SAMPLES, XOR_FEATURES);
    MtLinear* l1 = mt_linear_create(tape, XOR_FEATURES, XOR_HIDDEN, 1);
    MtLinear* l2 = mt_linear_create(tape, XOR_HIDDEN, 1, 1);
    MtSequential* model = mt_sequential_create(XOR_HIDDEN);
    MtOptimizer* optim = mt_adam_create(0.05f, 0.9f, 0.999f, 1e-8f);

    CHECK("allocation tape", tape != NULL ? 1.0f : 0.0f, 1.0f);
    CHECK("allocation dataset", dataset != NULL ? 1.0f : 0.0f, 1.0f);
    CHECK("allocation modele", model != NULL ? 1.0f : 0.0f, 1.0f);

    if (!tape || !dataset || !batch || !l1 || !l2 || !model || !optim) {
        mt_optimizer_free(optim);
        mt_sequential_free(model);
        mt_linear_free(l2);
        mt_linear_free(l1);
        mt_batch_free(batch);
        mt_dataset_free(dataset);
        ag_tape_free(tape);
        return;
    }

    init_xor_model(tape, l1, l2);
    mt_sequential_add_linear(model, l1);
    mt_sequential_add_activation(model, MT_ACT_TANH);
    mt_sequential_add_linear(model, l2);
    mt_sequential_add_activation(model, MT_ACT_SIGMOID);
    mt_optimizer_add_sequential(optim, model);

    float first_loss = train_epoch(tape, model, optim, dataset, batch);
    float last_loss = first_loss;
    for (int epoch = 2; epoch <= 100; epoch++) {
        last_loss = train_epoch(tape, model, optim, dataset, batch);
    }

    CHECK("la perte baisse", last_loss < first_loss ? 1.0f : 0.0f, 1.0f);

    int correct = 0;
    for (int i = 0; i < XOR_SAMPLES; i++) {
        float prob = eval_prob(tape, model, dataset, i);
        int pred = prob >= 0.5f ? 1 : 0;
        int truth = mt_dataset_label(dataset, i) >= 0.5f ? 1 : 0;
        if (pred == truth) {
            correct++;
        }
    }

    CHECK("XOR classe 4/4", (float)correct, 4.0f);

    mt_optimizer_free(optim);
    mt_sequential_free(model);
    mt_linear_free(l2);
    mt_linear_free(l1);
    mt_batch_free(batch);
    mt_dataset_free(dataset);
    ag_tape_free(tape);
}

static void summary(void) {
    printf("\n==============================================\n");
    printf("  %d reussis  |  %d echoues  |  %d total\n",
           _passed, _failed, _passed + _failed);
    printf("==============================================\n");
}

int main(void) {
    printf("test_xor : suite complete\n");
    test_mlp_learns_xor();
    summary();
    return _failed == 0 ? 0 : 1;
}
