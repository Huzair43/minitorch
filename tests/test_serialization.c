#include <stdio.h>
#include <math.h>

#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"
#include "minitorch/serialization/serialization.h"

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

static void test_linear_save_load(void) {
    section("Linear : sauvegarde et chargement");

    AgTape *t = ag_tape_create();
    MtLinear *linear = mt_linear_create(t, 2, 1, 1);

    mt_linear_set_weight(t, linear, 0, 0, 1.25f);
    mt_linear_set_weight(t, linear, 0, 1, -0.75f);
    mt_linear_set_bias(t, linear, 0, 0.5f);

    CHECK("save linear", (float)mt_save_linear(t, linear, "test_linear_save.mt"), 1.0f);

    mt_linear_set_weight(t, linear, 0, 0, 0.0f);
    mt_linear_set_weight(t, linear, 0, 1, 0.0f);
    mt_linear_set_bias(t, linear, 0, 0.0f);

    CHECK("load linear", (float)mt_load_linear(t, linear, "test_linear_save.mt"), 1.0f);
    CHECK("w0 restaure", ag_data(t, mt_linear_weight(linear, 0, 0)), 1.25f);
    CHECK("w1 restaure", ag_data(t, mt_linear_weight(linear, 0, 1)), -0.75f);
    CHECK("b restaure", ag_data(t, mt_linear_bias(linear, 0)), 0.5f);

    remove("test_linear_save.mt");
    mt_linear_free(linear);
    ag_tape_free(t);
}

static void test_sequential_save_load(void) {
    section("Sequential : sauvegarde et chargement");

    AgTape *t = ag_tape_create();
    MtLinear *l1 = mt_linear_create(t, 2, 2, 1);
    MtLinear *l2 = mt_linear_create(t, 2, 1, 1);
    MtSequential *seq = mt_sequential_create(2);

    mt_linear_set_weight(t, l1, 0, 0, 1.0f);
    mt_linear_set_weight(t, l1, 0, 1, 2.0f);
    mt_linear_set_bias(t, l1, 0, 0.1f);
    mt_linear_set_weight(t, l1, 1, 0, 3.0f);
    mt_linear_set_weight(t, l1, 1, 1, 4.0f);
    mt_linear_set_bias(t, l1, 1, 0.2f);

    mt_linear_set_weight(t, l2, 0, 0, -1.0f);
    mt_linear_set_weight(t, l2, 0, 1, 0.5f);
    mt_linear_set_bias(t, l2, 0, -0.3f);

    mt_sequential_add_linear(seq, l1);
    mt_sequential_add_activation(seq, MT_ACT_TANH);
    mt_sequential_add_linear(seq, l2);

    CHECK("save sequential", (float)mt_save_sequential(t, seq, "test_seq_save.mt"), 1.0f);

    mt_linear_set_weight(t, l1, 0, 0, 0.0f);
    mt_linear_set_weight(t, l1, 1, 1, 0.0f);
    mt_linear_set_weight(t, l2, 0, 1, 0.0f);
    mt_linear_set_bias(t, l2, 0, 0.0f);

    CHECK("load sequential", (float)mt_load_sequential(t, seq, "test_seq_save.mt"), 1.0f);
    CHECK("l1 w00 restaure", ag_data(t, mt_linear_weight(l1, 0, 0)), 1.0f);
    CHECK("l1 w11 restaure", ag_data(t, mt_linear_weight(l1, 1, 1)), 4.0f);
    CHECK("l2 w01 restaure", ag_data(t, mt_linear_weight(l2, 0, 1)), 0.5f);
    CHECK("l2 b restaure", ag_data(t, mt_linear_bias(l2, 0)), -0.3f);

    remove("test_seq_save.mt");
    mt_sequential_free(seq);
    mt_linear_free(l2);
    mt_linear_free(l1);
    ag_tape_free(t);
}

int main(void) {
    printf("test_serialization : suite complete\n");

    test_linear_save_load();
    test_sequential_save_load();

    summary();
    return _failed == 0 ? 0 : 1;
}
