#include <stdio.h>
#include <math.h>
#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"

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

int main(void) {
    printf("test_nn : suite complete\n");

    test_linear_forward_and_backward();
    test_linear_step();
    test_sequential_sigmoid();
    test_bce_loss();

    summary();
    return _failed == 0 ? 0 : 1;
}
