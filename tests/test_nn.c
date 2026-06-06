#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"
#include "minitorch/train/trainer.h"

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

static void test_linear_forward_and_backward(void) {
    section("Linear : forward + backward");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 2, 1, 1);

    mt_linear_set_weight(t, linear, 0, 0, 2.0f);
    mt_linear_set_weight(t, linear, 0, 1, 3.0f);
    mt_linear_set_bias(t, linear, 0, 1.0f);

    AgVal input[2] = {
        ag_leaf(t, 4.0f),
        ag_leaf(t, 5.0f)
    };
    AgVal target[1] = {
        ag_leaf(t, 20.0f)
    };
    AgVal pred[1];

    mt_linear_forward(t, linear, input, pred);
    AgVal loss = mt_mse_loss(t, pred, target, 1);
    ag_backward(t, loss);

    CHECK("prediction = 24", ag_data(t, pred[0]), 24.0f);
    CHECK("loss = 16", ag_data(t, loss), 16.0f);
    CHECK("dL/dw0 = 32", ag_grad(t, mt_linear_weight(linear, 0, 0)), 32.0f);
    CHECK("dL/dw1 = 40", ag_grad(t, mt_linear_weight(linear, 0, 1)), 40.0f);
    CHECK("dL/db = 8", ag_grad(t, mt_linear_bias(linear, 0)), 8.0f);

    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_linear_step(void) {
    section("Linear : descente de gradient");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 1, 1, 1);

    mt_linear_set_weight(t, linear, 0, 0, 2.0f);
    mt_linear_set_bias(t, linear, 0, 0.0f);

    AgVal input[1] = {ag_leaf(t, 3.0f)};
    AgVal target[1] = {ag_leaf(t, 5.0f)};
    AgVal pred[1];

    mt_linear_forward(t, linear, input, pred);
    AgVal loss = mt_mse_loss(t, pred, target, 1);
    ag_backward(t, loss);
    mt_linear_step(t, linear, 0.1f);

    CHECK("w apres mise a jour", ag_data(t, mt_linear_weight(linear, 0, 0)), 1.4f);
    CHECK("b apres mise a jour", ag_data(t, mt_linear_bias(linear, 0)), -0.2f);

    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_sequential_sigmoid(void) {
    section("Sequential : Linear + Sigmoid");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 1, 1, 1);
    MtSequential *seq = mt_sequential_create(1);

    mt_linear_set_weight(t, linear, 0, 0, 2.0f);
    mt_linear_set_bias(t, linear, 0, 0.0f);
    mt_sequential_add_linear(seq, linear);
    mt_sequential_add_activation(seq, MT_ACT_SIGMOID);

    AgVal input[1] = {ag_leaf(t, 0.0f)};
    AgVal pred[1];
    int ok = mt_sequential_forward(t, seq, input, 1, pred, 1);

    CHECK("forward accepte", (float)ok, 1.0f);
    CHECK("sigmoid(0) = 0.5", ag_data(t, pred[0]), 0.5f);

    mt_sequential_free(seq);
    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_bce_loss(void) {
    section("BCE loss");

    AgTape *t = ag_tape_create();
    AgVal pred[1] = {ag_leaf(t, 0.5f)};
    AgVal target[1] = {ag_leaf(t, 1.0f)};
    AgVal loss = mt_bce_loss(t, pred, target, 1);

    CHECK("BCE(0.5, 1)", ag_data(t, loss), -logf(0.5f + 1e-7f));

    ag_tape_free(t);
}

static void test_linear_init_zeros(void) {
    section("Linear : initialisation zeros");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 2, 2, 1);

    mt_linear_set_weight(t, linear, 0, 0, 2.0f);
    mt_linear_set_weight(t, linear, 0, 1, -3.0f);
    mt_linear_set_weight(t, linear, 1, 0, 4.0f);
    mt_linear_set_weight(t, linear, 1, 1, -5.0f);
    mt_linear_set_bias(t, linear, 0, 1.0f);
    mt_linear_set_bias(t, linear, 1, -1.0f);

    mt_linear_init_zeros(t, linear);

    int ok = 1;
    for (int out = 0; out < 2; out++) {
        for (int in = 0; in < 2; in++) {
            if (ag_data(t, mt_linear_weight(linear, out, in)) != 0.0f) ok = 0;
        }
        if (ag_data(t, mt_linear_bias(linear, out)) != 0.0f) ok = 0;
    }

    CHECK("poids et biais a zero", (float)ok, 1.0f);

    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_linear_init_xavier_uniform(void) {
    section("Linear : initialisation Xavier uniform");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 3, 2, 1);

    srand(1);
    mt_linear_init_xavier_uniform(t, linear);

    float limit = sqrtf(6.0f / 5.0f);
    int ok = 1;
    for (int out = 0; out < 2; out++) {
        for (int in = 0; in < 3; in++) {
            float value = ag_data(t, mt_linear_weight(linear, out, in));
            if (value < -limit || value > limit) ok = 0;
        }
        if (ag_data(t, mt_linear_bias(linear, out)) != 0.0f) ok = 0;
    }

    CHECK("poids dans bornes Xavier", (float)ok, 1.0f);

    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_linear_init_he_uniform(void) {
    section("Linear : initialisation He uniform");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 4, 3, 1);

    srand(2);
    mt_linear_init_he_uniform(t, linear);

    float limit = sqrtf(6.0f / 4.0f);
    int ok = 1;
    for (int out = 0; out < 3; out++) {
        for (int in = 0; in < 4; in++) {
            float value = ag_data(t, mt_linear_weight(linear, out, in));
            if (value < -limit || value > limit) ok = 0;
        }
        if (ag_data(t, mt_linear_bias(linear, out)) != 0.0f) ok = 0;
    }

    CHECK("poids dans bornes He", (float)ok, 1.0f);

    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_metrics_argmax_and_mean(void) {
    section("Métriques : argmax et moyenne");

    AgTape *t = ag_tape_create();
    AgVal values[3] = {
        ag_leaf(t, -1.0f),
        ag_leaf(t, 3.0f),
        ag_leaf(t, 2.0f)
    };
    float losses[3] = {1.0f, 2.0f, 4.0f};

    CHECK("argmax AgVal = 1", (float)mt_argmax(t, values, 3), 1.0f);
    CHECK("moyenne = 7/3", mt_mean(losses, 3), 7.0f / 3.0f);

    ag_tape_free(t);
}

static void test_metrics_accuracy(void) {
    section("Métriques : exactitude");

    float probs[4] = {0.1f, 0.8f, 0.7f, 0.4f};
    float binary_target[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    int pred[5] = {0, 2, 1, 1, 2};
    int target[5] = {0, 1, 1, 1, 2};

    CHECK("exactitude binaire = 3/4", mt_accuracy_binary(probs, binary_target, 4, 0.5f), 0.75f);
    CHECK("exactitude multi = 4/5", mt_accuracy_multiclass(pred, target, 5), 0.8f);
}

static void test_metrics_confusion_matrix(void) {
    section("Métriques : matrice de confusion");

    int pred[5] = {0, 2, 1, 1, 2};
    int target[5] = {0, 1, 1, 1, 2};
    int matrix[9];

    mt_confusion_matrix(pred, target, 5, 3, matrix);

    CHECK("confusion[0,0]", (float)matrix[0], 1.0f);
    CHECK("confusion[1,1]", (float)matrix[4], 2.0f);
    CHECK("confusion[1,2]", (float)matrix[5], 1.0f);
    CHECK("confusion[2,2]", (float)matrix[8], 1.0f);
}

static void test_eval_binary_linear(void) {
    section("Évaluation : Linear binaire");

    float x[2] = {0.0f, 1.0f};
    float y[2] = {0.0f, 1.0f};

    AgTape *t = ag_tape_create();
    MtDataset *dataset = mt_dataset_create(x, y, 2, 1);
    MtLinear *model = mt_linear_create(t, 1, 1, 1);

    mt_linear_set_weight(t, model, 0, 0, 10.0f);
    mt_linear_set_bias(t, model, 0, -5.0f);
    int graph_checkpoint = ag_checkpoint(t);

    MtEvalResult eval;
    int ok = mt_eval_binary_linear(t, model, graph_checkpoint, dataset, 0.5f, &eval);

    CHECK("eval binaire accepte", (float)ok, 1.0f);
    CHECK("eval binaire total", (float)eval.total, 2.0f);
    CHECK("eval binaire correct", (float)eval.correct, 2.0f);
    CHECK("eval binaire exactitude", eval.accuracy, 1.0f);
    CHECK("eval binaire perte basse", eval.loss_mean < 0.01f ? 1.0f : 0.0f, 1.0f);

    mt_linear_free(model);
    mt_dataset_free(dataset);
    ag_tape_free(t);
}

static void test_eval_multiclass_linear(void) {
    section("Évaluation : Linear multi-classe");

    float x[3] = {-2.0f, 0.0f, 2.0f};
    float y[3] = {0.0f, 1.0f, 2.0f};

    AgTape *t = ag_tape_create();
    MtDataset *dataset = mt_dataset_create(x, y, 3, 1);
    MtLinear *model = mt_linear_create(t, 1, 3, 1);

    mt_linear_set_weight(t, model, 0, 0, -1.0f);
    mt_linear_set_bias(t, model, 0, 0.0f);
    mt_linear_set_weight(t, model, 1, 0, 0.0f);
    mt_linear_set_bias(t, model, 1, 1.0f);
    mt_linear_set_weight(t, model, 2, 0, 1.0f);
    mt_linear_set_bias(t, model, 2, 0.0f);
    int graph_checkpoint = ag_checkpoint(t);

    MtEvalResult eval;
    int confusion[9];
    int ok = mt_eval_multiclass_linear(t, model, graph_checkpoint, dataset, 3, &eval, confusion);

    CHECK("eval multi accepte", (float)ok, 1.0f);
    CHECK("eval multi total", (float)eval.total, 3.0f);
    CHECK("eval multi correct", (float)eval.correct, 3.0f);
    CHECK("eval multi exactitude", eval.accuracy, 1.0f);
    CHECK("eval multi confusion[0,0]", (float)confusion[0], 1.0f);
    CHECK("eval multi confusion[1,1]", (float)confusion[4], 1.0f);
    CHECK("eval multi confusion[2,2]", (float)confusion[8], 1.0f);

    mt_linear_free(model);
    mt_dataset_free(dataset);
    ag_tape_free(t);
}

static void test_model_linear_binary_eval(void) {
    section("MtModel : Linear binaire + évaluation");

    float x[2] = {0.0f, 1.0f};
    float y[2] = {0.0f, 1.0f};

    AgTape *t = ag_tape_create();
    MtDataset *dataset = mt_dataset_create(x, y, 2, 1);
    MtModel *model = mt_model_create_linear_binary(t, 1);

    mt_linear_set_weight(t, model->linear, 0, 0, 10.0f);
    mt_linear_set_bias(t, model->linear, 0, -5.0f);
    int graph_checkpoint = ag_checkpoint(t);

    MtEvalResult eval;
    int ok = mt_model_eval_binary(t, model, graph_checkpoint, dataset, 0.5f, &eval);

    CHECK("model eval accepte", (float)ok, 1.0f);
    CHECK("model eval exactitude", eval.accuracy, 1.0f);
    CHECK("model eval correct", (float)eval.correct, 2.0f);

    mt_model_free(model);
    mt_dataset_free(dataset);
    ag_tape_free(t);
}

static void test_model_mlp_forward(void) {
    section("MtModel : MLP forward");

    AgTape *t = ag_tape_create();
    MtModel *model = mt_model_create_mlp_binary(t, 2, 4);
    AgVal input[2] = {
        ag_leaf(t, 0.25f),
        ag_leaf(t, -0.50f)
    };
    AgVal output[1];

    int ok = mt_model_forward(t, model, input, 2, output, 1);
    float prob = ag_data(t, output[0]);

    CHECK("mlp forward accepte", (float)ok, 1.0f);
    CHECK("mlp proba dans [0, 1]", (prob >= 0.0f && prob <= 1.0f) ? 1.0f : 0.0f, 1.0f);

    mt_model_free(model);
    ag_tape_free(t);
}

static void test_model_save_load_linear(void) {
    section("MtModel : sauvegarde et chargement");

    const char *path = "test_model_linear.mt";
    AgTape *t = ag_tape_create();
    MtModel *model = mt_model_create_linear_binary(t, 2);

    mt_linear_set_weight(t, model->linear, 0, 0, 1.25f);
    mt_linear_set_weight(t, model->linear, 0, 1, -0.75f);
    mt_linear_set_bias(t, model->linear, 0, 0.50f);

    int saved = mt_model_save(t, model, path);
    mt_linear_set_weight(t, model->linear, 0, 0, 0.0f);
    mt_linear_set_weight(t, model->linear, 0, 1, 0.0f);
    mt_linear_set_bias(t, model->linear, 0, 0.0f);
    int loaded = mt_model_load(t, model, path);

    CHECK("model sauvegarde accepte", (float)saved, 1.0f);
    CHECK("model chargement accepte", (float)loaded, 1.0f);
    CHECK("model w0 restauré", ag_data(t, mt_linear_weight(model->linear, 0, 0)), 1.25f);
    CHECK("model w1 restauré", ag_data(t, mt_linear_weight(model->linear, 0, 1)), -0.75f);
    CHECK("model biais restauré", ag_data(t, mt_linear_bias(model->linear, 0)), 0.50f);

    remove(path);
    mt_model_free(model);
    ag_tape_free(t);
}

static void test_loss_abstraction(void) {
    section("MtLoss : abstraction");

    AgTape *t = ag_tape_create();
    MtLoss loss = mt_loss_create(MT_LOSS_BCE);
    AgVal pred[1] = {ag_leaf(t, 0.5f)};
    AgVal target[1] = {ag_leaf(t, 1.0f)};
    AgVal value = mt_loss_forward(t, &loss, pred, target, 1);

    CHECK("loss BCE via MtLoss", ag_data(t, value), -logf(0.5f + 1e-7f));

    ag_tape_free(t);
}

static void test_model_predict_api(void) {
    section("MtModel : predict");

    AgTape *t = ag_tape_create();
    MtModel *model = mt_model_create_linear_binary(t, 1);
    mt_linear_set_weight(t, model->linear, 0, 0, 10.0f);
    mt_linear_set_bias(t, model->linear, 0, -5.0f);
    int graph_checkpoint = ag_checkpoint(t);

    float features[1] = {1.0f};
    float output[1] = {0.0f};
    int ok = mt_model_predict(t, model, graph_checkpoint, features, output, 1);

    CHECK("predict accepte", (float)ok, 1.0f);
    CHECK("predict classe positive", output[0] > 0.5f ? 1.0f : 0.0f, 1.0f);

    mt_model_free(model);
    ag_tape_free(t);
}

static void test_trainer_binary_api(void) {
    section("MtTrainer : entraînement binaire");

    float x[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    float y[4] = {0.0f, 0.0f, 1.0f, 1.0f};

    AgTape *t = ag_tape_create();
    MtDataset *dataset = mt_dataset_create(x, y, 4, 1);
    MtModel *model = mt_model_create_linear_binary(t, 1);
    MtOptimizer *optim = mt_sgd_create(0.05f);
    mt_optimizer_add_linear(optim, model->linear);

    int graph_checkpoint = ag_checkpoint(t);
    MtLoss loss = mt_loss_create(MT_LOSS_BCE);
    MtTrainConfig config = mt_train_config_default();
    config.epochs = 2;
    config.batch_size = 2;
    config.shuffle = 0;

    MtTrainHistory history;
    int ok = mt_trainer_train_binary(t, model, optim, &loss, graph_checkpoint, dataset, &config, &history);

    CHECK("trainer accepte", (float)ok, 1.0f);
    CHECK("trainer époques", (float)history.epochs_ran, 2.0f);
    CHECK("trainer samples vus", (float)history.samples_seen, 8.0f);

    mt_optimizer_free(optim);
    mt_model_free(model);
    mt_dataset_free(dataset);
    ag_tape_free(t);
}

int main(void) {
    printf("test_nn : suite complete\n");

    test_linear_forward_and_backward();
    test_linear_step();
    test_sequential_sigmoid();
    test_bce_loss();
    test_linear_init_zeros();
    test_linear_init_xavier_uniform();
    test_linear_init_he_uniform();
    test_metrics_argmax_and_mean();
    test_metrics_accuracy();
    test_metrics_confusion_matrix();
    test_eval_binary_linear();
    test_eval_multiclass_linear();
    test_model_linear_binary_eval();
    test_model_mlp_forward();
    test_model_save_load_linear();
    test_loss_abstraction();
    test_model_predict_api();
    test_trainer_binary_api();

    summary();
    return _failed == 0 ? 0 : 1;
}
