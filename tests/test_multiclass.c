#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

#define N_CLASSES 3

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

static void section(const char *title) {
    printf("\n-- %s\n", title);
}

static void summary(void) {
    printf("\n==============================================\n");
    printf("  %d reussis  |  %d echoues  |  %d total\n",
           _passed, _failed, _passed + _failed);
    printf("==============================================\n");
}

static int argmax(AgTape* tape, const AgVal* probs, int n) {
    int best = 0;
    float best_value = ag_data(tape, probs[0]);
    for (int i = 1; i < n; i++) {
        float value = ag_data(tape, probs[i]);
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return best;
}

static void test_softmax_distribution(void) {
    section("Softmax : distribution");

    AgTape* tape = ag_tape_create();
    AgVal logits[3] = {
        ag_leaf(tape, 1.0f),
        ag_leaf(tape, 2.0f),
        ag_leaf(tape, 3.0f)
    };
    AgVal probs[3];

    mt_softmax(tape, logits, 3, probs);

    float sum = ag_data(tape, probs[0]) + ag_data(tape, probs[1]) + ag_data(tape, probs[2]);
    CHECK("somme softmax = 1", sum, 1.0f);
    CHECK("classe max = 2", (float)argmax(tape, probs, 3), 2.0f);

    ag_tape_free(tape);
}

static void test_cross_entropy_value(void) {
    section("CrossEntropy : valeur");

    AgTape* tape = ag_tape_create();
    AgVal probs[3] = {
        ag_leaf(tape, 0.1f),
        ag_leaf(tape, 0.7f),
        ag_leaf(tape, 0.2f)
    };
    AgVal target[3] = {
        ag_leaf(tape, 0.0f),
        ag_leaf(tape, 1.0f),
        ag_leaf(tape, 0.0f)
    };

    AgVal loss = mt_cross_entropy_loss(tape, probs, target, 3);
    CHECK("CE = -log(0.7)", ag_data(tape, loss), -logf(0.7f + 1e-7f));

    ag_tape_free(tape);
}

static void test_cross_entropy_from_logits_value(void) {
    section("CrossEntropyFromLogits : valeur stable");

    AgTape* tape = ag_tape_create();
    AgVal logits[3] = {
        ag_leaf(tape, 1.0f),
        ag_leaf(tape, 2.0f),
        ag_leaf(tape, 3.0f)
    };
    AgVal target[3] = {
        ag_leaf(tape, 0.0f),
        ag_leaf(tape, 0.0f),
        ag_leaf(tape, 1.0f)
    };

    AgVal loss = mt_cross_entropy_from_logits(tape, logits, target, 3);
    float expected = logf(expf(1.0f - 3.0f) + expf(2.0f - 3.0f) + 1.0f);
    CHECK("CE logits classe 2", ag_data(tape, loss), expected);

    ag_backward(tape, loss);
    CHECK("gradient logit cible negatif", ag_grad(tape, logits[2]) < 0.0f ? 1.0f : 0.0f, 1.0f);

    ag_tape_free(tape);
}

static float train_one_epoch(AgTape* tape,
                             MtLinear* model,
                             MtOptimizer* optim,
                             int graph_checkpoint,
                             MtDataset* dataset,
                             MtBatch* batch) {
    ag_rewind(tape, graph_checkpoint);

    int count = mt_dataset_get_batch(dataset, 0, 6, batch);
    AgVal losses[6];

    for (int i = 0; i < count; i++) {
        AgVal input[2] = {
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

static void test_multiclass_training(void) {
    section("Training : classification 3 classes");

    float x[] = {
        -2.0f, -2.0f,
        -1.0f, -2.0f,
         2.0f,  0.0f,
         3.0f,  0.0f,
         0.0f,  2.0f,
         0.0f,  3.0f
    };
    float y[] = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f};

    AgTape* tape = ag_tape_create();
    MtDataset* dataset = mt_dataset_create(x, y, 6, 2);
    MtBatch* batch = mt_batch_create(6, 2);
    MtLinear* model = mt_linear_create(tape, 2, N_CLASSES, 1);
    MtOptimizer* optim = mt_adam_create(0.05f, 0.9f, 0.999f, 1e-8f);

    mt_linear_set_weight(tape, model, 0, 0, -0.5f);
    mt_linear_set_weight(tape, model, 0, 1, -0.5f);
    mt_linear_set_bias(tape, model, 0, 0.0f);
    mt_linear_set_weight(tape, model, 1, 0, 0.5f);
    mt_linear_set_weight(tape, model, 1, 1, 0.0f);
    mt_linear_set_bias(tape, model, 1, 0.0f);
    mt_linear_set_weight(tape, model, 2, 0, 0.0f);
    mt_linear_set_weight(tape, model, 2, 1, 0.5f);
    mt_linear_set_bias(tape, model, 2, 0.0f);

    mt_optimizer_add_linear(optim, model);
    int graph_checkpoint = ag_checkpoint(tape);

    float first_loss = train_one_epoch(tape, model, optim, graph_checkpoint, dataset, batch);
    float last_loss = first_loss;
    for (int epoch = 2; epoch <= 100; epoch++) {
        last_loss = train_one_epoch(tape, model, optim, graph_checkpoint, dataset, batch);
    }

    CHECK("la perte baisse", last_loss < first_loss ? 1.0f : 0.0f, 1.0f);

    int correct = 0;
    for (int i = 0; i < dataset->n_samples; i++) {
        ag_rewind(tape, graph_checkpoint);
        AgVal input[2] = {
            ag_leaf(tape, mt_dataset_feature(dataset, i, 0)),
            ag_leaf(tape, mt_dataset_feature(dataset, i, 1))
        };
        AgVal logits[N_CLASSES];
        AgVal probs[N_CLASSES];
        mt_linear_forward(tape, model, input, logits);
        mt_softmax(tape, logits, N_CLASSES, probs);
        int pred = argmax(tape, probs, N_CLASSES);
        int truth = (int)mt_dataset_label(dataset, i);
        if (pred == truth) correct++;
    }

    CHECK("classification 6/6", (float)correct, 6.0f);

    mt_optimizer_free(optim);
    mt_linear_free(model);
    mt_batch_free(batch);
    mt_dataset_free(dataset);
    ag_tape_free(tape);
}

int main(void) {
    printf("test_multiclass : suite complete\n");

    test_softmax_distribution();
    test_cross_entropy_value();
    test_cross_entropy_from_logits_value();
    test_multiclass_training();

    summary();
    return _failed == 0 ? 0 : 1;
}
