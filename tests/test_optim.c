#include <stdio.h>
#include <math.h>
#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

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

static AgVal build_square_loss(AgTape *t, AgVal param) {
    AgVal target = ag_leaf(t, 3.0f);
    AgVal diff = ag_sub(t, param, target);
    return ag_square(t, diff);
}

static void test_sgd_step(void) {
    section("SGD : mise a jour simple");

    AgTape *t = ag_tape_create();
    MtOptimizer *opt = mt_sgd_create(0.1f);
    AgVal w = ag_leaf(t, 5.0f);
    mt_optimizer_add_param(opt, w);

    AgVal loss = build_square_loss(t, w);
    mt_optimizer_zero_grad(t, opt);
    ag_backward(t, loss);
    mt_optimizer_step(t, opt);

    CHECK("gradient = 4", ag_grad(t, w), 4.0f);
    CHECK("w apres SGD", ag_data(t, w), 4.6f);

    mt_optimizer_free(opt);
    ag_tape_free(t);
}

static void test_momentum_step(void) {
    section("Momentum : accumulation de vitesse");

    AgTape *t = ag_tape_create();
    MtOptimizer *opt = mt_momentum_create(0.1f, 0.9f);
    AgVal w = ag_leaf(t, 5.0f);
    mt_optimizer_add_param(opt, w);

    AgVal loss1 = build_square_loss(t, w);
    mt_optimizer_zero_grad(t, opt);
    ag_backward(t, loss1);
    mt_optimizer_step(t, opt);

    AgVal loss2 = build_square_loss(t, w);
    mt_optimizer_zero_grad(t, opt);
    ag_backward(t, loss2);
    mt_optimizer_step(t, opt);

    CHECK("w apres deux steps momentum", ag_data(t, w), 3.92f);

    mt_optimizer_free(opt);
    ag_tape_free(t);
}

static void test_adam_step(void) {
    section("Adam : premier step corrige");

    AgTape *t = ag_tape_create();
    MtOptimizer *opt = mt_adam_create(0.1f, 0.9f, 0.999f, 1e-8f);
    AgVal w = ag_leaf(t, 5.0f);
    mt_optimizer_add_param(opt, w);

    AgVal loss = build_square_loss(t, w);
    mt_optimizer_zero_grad(t, opt);
    ag_backward(t, loss);
    mt_optimizer_step(t, opt);

    CHECK("w apres Adam", ag_data(t, w), 4.9f);

    mt_optimizer_free(opt);
    ag_tape_free(t);
}

static void test_add_linear_params(void) {
    section("Optim : parametres de Linear");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 2, 1, 1);
    MtOptimizer *opt = mt_sgd_create(0.1f);

    mt_linear_set_weight(t, linear, 0, 0, 1.0f);
    mt_linear_set_weight(t, linear, 0, 1, 2.0f);
    mt_linear_set_bias(t, linear, 0, 0.5f);
    mt_optimizer_add_linear(opt, linear);

    AgVal input[2] = {
        ag_leaf(t, 1.0f),
        ag_leaf(t, 1.0f)
    };
    AgVal target[1] = {
        ag_leaf(t, 0.0f)
    };
    AgVal pred[1];

    mt_linear_forward(t, linear, input, pred);
    AgVal loss = mt_mse_loss(t, pred, target, 1);
    mt_optimizer_zero_grad(t, opt);
    ag_backward(t, loss);
    mt_optimizer_step(t, opt);

    CHECK("w0 mis a jour", ag_data(t, mt_linear_weight(linear, 0, 0)), 0.3f);
    CHECK("w1 mis a jour", ag_data(t, mt_linear_weight(linear, 0, 1)), 1.3f);
    CHECK("b mis a jour", ag_data(t, mt_linear_bias(linear, 0)), -0.2f);

    mt_optimizer_free(opt);
    mt_linear_free(linear);
    ag_tape_free(t);
}

int main(void) {
    printf("test_optim : suite complete\n");

    test_sgd_step();
    test_momentum_step();
    test_adam_step();
    test_add_linear_params();

    summary();
    return _failed == 0 ? 0 : 1;
}
