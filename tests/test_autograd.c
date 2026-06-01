#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minitorch/core/autograd.h"

/* ════════════════════════════════════════════════════════
   FRAMEWORK DE TEST MINIMAL
   ════════════════════════════════════════════════════════ */

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
    printf("\n── %s\n", title);
}

static void summary(void) {
    printf("\n════════════════════════════════════════════════\n");
    printf("  %d passed  |  %d failed  |  %d total\n",
           _passed, _failed, _passed + _failed);
    printf("════════════════════════════════════════════════\n");
}

/* ════════════════════════════════════════════════════════
   TESTS OPS SCALAIRES
   ════════════════════════════════════════════════════════ */

static void test_add(void) {
    section("add : z = a + b");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 3.0f);
    AgVal b = ag_leaf(t, 5.0f);
    AgVal z = ag_add(t, a, b);
    ag_backward(t, z);
    CHECK("forward  z = 3+5",   ag_data(t, z), 8.0f);
    CHECK("backward dz/da = 1", ag_grad(t, a), 1.0f);
    CHECK("backward dz/db = 1", ag_grad(t, b), 1.0f);
    ag_tape_free(t);
}

static void test_sub(void) {
    section("sub : z = a - b");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 7.0f);
    AgVal b = ag_leaf(t, 2.0f);
    AgVal z = ag_sub(t, a, b);
    ag_backward(t, z);
    CHECK("forward  z = 7-2",    ag_data(t, z), 5.0f);
    CHECK("backward dz/da =  1", ag_grad(t, a),  1.0f);
    CHECK("backward dz/db = -1", ag_grad(t, b), -1.0f);
    ag_tape_free(t);
}

static void test_mul(void) {
    section("mul : z = a * b");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 4.0f);
    AgVal b = ag_leaf(t, 3.0f);
    AgVal z = ag_mul(t, a, b);
    ag_backward(t, z);
    CHECK("forward  z = 4*3",   ag_data(t, z), 12.0f);
    CHECK("backward dz/da = b", ag_grad(t, a),  3.0f);
    CHECK("backward dz/db = a", ag_grad(t, b),  4.0f);
    ag_tape_free(t);
}

static void test_div(void) {
    section("div : z = a / b");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 6.0f);
    AgVal b = ag_leaf(t, 2.0f);
    AgVal z = ag_div(t, a, b);
    ag_backward(t, z);
    /* dz/da = 1/b = 0.5,  dz/db = -a/b^2 = -6/4 = -1.5 */
    CHECK("forward  z = 6/2",       ag_data(t, z),  3.0f);
    CHECK("backward dz/da = 1/b",   ag_grad(t, a),  0.5f);
    CHECK("backward dz/db = -a/b2", ag_grad(t, b), -1.5f);
    ag_tape_free(t);
}

static void test_neg(void) {
    section("neg : z = -a");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 5.0f);
    AgVal z = ag_neg(t, a);
    ag_backward(t, z);
    CHECK("forward  z = -5",    ag_data(t, z), -5.0f);
    CHECK("backward dz/da = -1", ag_grad(t, a), -1.0f);
    ag_tape_free(t);
}

static void test_pow(void) {
    section("pow : z = a^3");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 2.0f);
    AgVal z = ag_pow(t, a, 3.0f);
    ag_backward(t, z);
    /* dz/da = 3*a^2 = 3*4 = 12 */
    CHECK("forward  z = 2^3",       ag_data(t, z),  8.0f);
    CHECK("backward dz/da = 3*a^2", ag_grad(t, a), 12.0f);
    ag_tape_free(t);
}

static void test_relu(void) {
    section("relu : actif et eteint");
    {
        AgTape *t = ag_tape_create();
        AgVal a = ag_leaf(t, 3.0f);
        AgVal z = ag_relu(t, a);
        ag_backward(t, z);
        CHECK("relu( 3) forward  = 3", ag_data(t, z), 3.0f);
        CHECK("relu( 3) backward = 1", ag_grad(t, a), 1.0f);
        ag_tape_free(t);
    }
    {
        AgTape *t = ag_tape_create();
        AgVal a = ag_leaf(t, -2.0f);
        AgVal z = ag_relu(t, a);
        ag_backward(t, z);
        CHECK("relu(-2) forward  = 0", ag_data(t, z), 0.0f);
        CHECK("relu(-2) backward = 0", ag_grad(t, a), 0.0f);
        ag_tape_free(t);
    }
}

static void test_tanh(void) {
    section("tanh : z = tanh(a)");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 1.0f);
    AgVal z = ag_tanh(t, a);
    ag_backward(t, z);
    float expected_fwd = tanhf(1.0f);
    float expected_bwd = 1.0f - expected_fwd * expected_fwd;
    CHECK("tanh(1) forward",  ag_data(t, z), expected_fwd);
    CHECK("tanh(1) backward", ag_grad(t, a), expected_bwd);
    ag_tape_free(t);
}

static void test_sigmoid(void) {
    section("sigmoid : z = sigmoid(a)");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 0.0f);
    AgVal z = ag_sigmoid(t, a);
    ag_backward(t, z);
    /* sigmoid(0) = 0.5,  grad = 0.5*(1-0.5) = 0.25 */
    CHECK("sigmoid(0) forward  = 0.5",  ag_data(t, z), 0.5f);
    CHECK("sigmoid(0) backward = 0.25", ag_grad(t, a), 0.25f);
    ag_tape_free(t);
}

static void test_exp(void) {
    section("exp : z = e^a");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 1.0f);
    AgVal z = ag_exp(t, a);
    ag_backward(t, z);
    CHECK("exp(1) forward  = e",   ag_data(t, z), expf(1.0f));
    CHECK("exp(1) backward = e",   ag_grad(t, a), expf(1.0f));
    ag_tape_free(t);
}

static void test_log(void) {
    section("log : z = ln(a)");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 2.0f);
    AgVal z = ag_log(t, a);
    ag_backward(t, z);
    /* dz/da = 1/a = 0.5 */
    CHECK("log(2) forward  = ln2", ag_data(t, z), logf(2.0f));
    CHECK("log(2) backward = 0.5", ag_grad(t, a), 0.5f);
    ag_tape_free(t);
}

/* ════════════════════════════════════════════════════════
   TESTS CHAIN RULE
   ════════════════════════════════════════════════════════ */

static void test_chain_rule(void) {
    section("chain rule : z = (a + b) * c");
    /*
     * z = (a+b)*c
     * dz/da = c = 4
     * dz/db = c = 4
     * dz/dc = a+b = 5
     */
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 2.0f);
    AgVal b = ag_leaf(t, 3.0f);
    AgVal c = ag_leaf(t, 4.0f);
    AgVal s = ag_add(t, a, b);
    AgVal z = ag_mul(t, s, c);
    ag_backward(t, z);
    CHECK("forward  z = 20",   ag_data(t, z), 20.0f);
    CHECK("backward dz/da = 4", ag_grad(t, a),  4.0f);
    CHECK("backward dz/db = 4", ag_grad(t, b),  4.0f);
    CHECK("backward dz/dc = 5", ag_grad(t, c),  5.0f);
    ag_tape_free(t);
}

static void test_shared_node(void) {
    section("noeud partage : z = a * a  (gradient accumule)");
    /*
     * z = a * a = a^2
     * dz/da = 2*a = 6
     * Le noeud 'a' est utilise deux fois -> accumulation
     */
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 3.0f);
    AgVal z = ag_mul(t, a, a);
    ag_backward(t, z);
    CHECK("forward  z = 9",     ag_data(t, z), 9.0f);
    CHECK("backward dz/da = 6", ag_grad(t, a), 6.0f);
    ag_tape_free(t);
}

static void test_mse_loss(void) {
    section("MSE loss : L = (y_hat - y)^2");
    /*
     * y_hat = x*w + b = 2*3 + 1 = 7
     * L = (7 - 5)^2 = 4
     * dL/dw = 2*(y_hat - y)*x = 2*2*2 = 8
     * dL/db = 2*(y_hat - y)   = 2*2   = 4
     */
    AgTape *t = ag_tape_create();
    AgVal x = ag_leaf(t, 2.0f);
    AgVal w = ag_leaf(t, 3.0f);
    AgVal b = ag_leaf(t, 1.0f);
    AgVal y = ag_leaf(t, 5.0f);

    AgVal y_hat = ag_add(t, ag_mul(t, x, w), b);
    AgVal diff  = ag_sub(t, y_hat, y);
    AgVal loss  = ag_mul(t, diff, diff);

    ag_backward(t, loss);

    CHECK("forward  y_hat = 7",  ag_data(t, y_hat), 7.0f);
    CHECK("forward  loss  = 4",  ag_data(t, loss),  4.0f);
    CHECK("backward dL/dw = 8",  ag_grad(t, w),     8.0f);
    CHECK("backward dL/db = 4",  ag_grad(t, b),     4.0f);
    ag_tape_free(t);
}

/* ════════════════════════════════════════════════════════
   TESTS OPS TENSORIELLES
   ════════════════════════════════════════════════════════ */

static void test_sum(void) {
    section("sum : z = a + b + c + d");
    AgTape *t = ag_tape_create();
    AgVal a = ag_leaf(t, 1.0f);
    AgVal b = ag_leaf(t, 2.0f);
    AgVal c = ag_leaf(t, 3.0f);
    AgVal d = ag_leaf(t, 4.0f);
    AgVal vals[] = {a, b, c, d};
    AgVal z = ag_sum(t, vals, 4);
    ag_backward(t, z);
    CHECK("forward  z = 10",    ag_data(t, z), 10.0f);
    CHECK("backward dz/da = 1", ag_grad(t, a),  1.0f);
    CHECK("backward dz/db = 1", ag_grad(t, b),  1.0f);
    CHECK("backward dz/dc = 1", ag_grad(t, c),  1.0f);
    CHECK("backward dz/dd = 1", ag_grad(t, d),  1.0f);
    ag_tape_free(t);
}

static void test_matmul(void) {
    section("matmul : A(2x2) @ B(2x2) = C(2x2)");
    /*
     * A = [[1, 2],    B = [[5, 6],    C = A@B = [[19, 22],
     *      [3, 4]]         [7, 8]]               [43, 50]]
     *
     * loss = sum(C) = 19+22+43+50 = 134
     * dL/dA[i,j] = sum_k dL/dC[i,k] * B[j,k]  (B transposee)
     *            = sum_k 1 * B[j,k]
     * dL/dA = [[5+6, 7+8],   = [[11, 15],
     *          [5+6, 7+8]]      [11, 15]]
     * dL/dB[i,j] = sum_k A[k,i] * dL/dC[k,j]
     *            = sum_k A[k,i] * 1
     * dL/dB = [[1+3, 1+3],   = [[4, 4],
     *          [2+4, 2+4]]      [6, 6]]
     */
    AgTape *t = ag_tape_create();

    AgVal A[4], B[4], C[4];
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    for (int i = 0; i < 4; i++) {
        A[i] = ag_leaf(t, a_data[i]);
        B[i] = ag_leaf(t, b_data[i]);
    }

    ag_matmul(t, A, 2, 2, B, 2, C);

    AgVal all[] = {C[0], C[1], C[2], C[3]};
    AgVal loss  = ag_sum(t, all, 4);

    ag_backward(t, loss);

    CHECK("C[0,0] = 19", ag_data(t, C[0]), 19.0f);
    CHECK("C[0,1] = 22", ag_data(t, C[1]), 22.0f);
    CHECK("C[1,0] = 43", ag_data(t, C[2]), 43.0f);
    CHECK("C[1,1] = 50", ag_data(t, C[3]), 50.0f);
    CHECK("dL/dA[0,0] = 11", ag_grad(t, A[0]), 11.0f);
    CHECK("dL/dA[0,1] = 15", ag_grad(t, A[1]), 15.0f);
    CHECK("dL/dA[1,0] = 11", ag_grad(t, A[2]), 11.0f);
    CHECK("dL/dA[1,1] = 15", ag_grad(t, A[3]), 15.0f);
    CHECK("dL/dB[0,0] =  4", ag_grad(t, B[0]),  4.0f);
    CHECK("dL/dB[0,1] =  4", ag_grad(t, B[1]),  4.0f);
    CHECK("dL/dB[1,0] =  6", ag_grad(t, B[2]),  6.0f);
    CHECK("dL/dB[1,1] =  6", ag_grad(t, B[3]),  6.0f);

    ag_tape_free(t);
}

static void test_transpose(void) {
    section("transpose : A(2x3) -> At(3x2)");
    /*
     * A = [[1, 2, 3],    At = [[1, 4],
     *      [4, 5, 6]]          [2, 5],
     *                          [3, 6]]
     */
    AgTape *t = ag_tape_create();
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    AgVal A[6], At[6];
    for (int i = 0; i < 6; i++)
        A[i] = ag_leaf(t, a_data[i]);

    ag_transpose(t, A, 2, 3, At);

    /* At est une reindexation des memes AgVal, pas de nouveaux noeuds */
    CHECK("At[0,0] = A[0,0] = 1", ag_data(t, At[0]), 1.0f);
    CHECK("At[1,0] = A[0,1] = 2", ag_data(t, At[1]), 2.0f);
    CHECK("At[2,0] = A[0,2] = 3", ag_data(t, At[2]), 3.0f);
    CHECK("At[0,1] = A[1,0] = 4", ag_data(t, At[3]), 4.0f);
    CHECK("At[1,1] = A[1,1] = 5", ag_data(t, At[4]), 5.0f);
    CHECK("At[2,1] = A[1,2] = 6", ag_data(t, At[5]), 6.0f);

    ag_tape_free(t);
}

/* ════════════════════════════════════════════════════════
   TEST MEMOIRE : reset et reutilisation du tape
   ════════════════════════════════════════════════════════ */

static void test_tape_reset(void) {
    section("tape reset : reutilisation apres ag_reset");
    AgTape *t = ag_tape_create();

    AgVal a = ag_leaf(t, 2.0f);
    AgVal b = ag_leaf(t, 3.0f);
    AgVal z = ag_mul(t, a, b);
    ag_backward(t, z);
    CHECK("avant reset : z = 6", ag_data(t, z), 6.0f);

    ag_reset(t);

    /* recree les memes noeuds, indices repartent de 0 */
    a = ag_leaf(t, 5.0f);
    b = ag_leaf(t, 4.0f);
    z = ag_mul(t, a, b);
    ag_backward(t, z);
    CHECK("apres reset : z = 20",   ag_data(t, z), 20.0f);
    CHECK("apres reset : dz/da = 4", ag_grad(t, a),  4.0f);

    ag_tape_free(t);
}

static void test_zero_grad(void) {
    section("zero_grad : les gradients sont remis a zero");
    AgTape *t = ag_tape_create();

    AgVal a = ag_leaf(t, 3.0f);
    AgVal b = ag_leaf(t, 4.0f);
    AgVal z = ag_mul(t, a, b);
    ag_backward(t, z);
    CHECK("avant zero_grad : grad(a) = 4", ag_grad(t, a), 4.0f);

    ag_zero_grad(t);
    CHECK("apres zero_grad : grad(a) = 0", ag_grad(t, a), 0.0f);
    CHECK("apres zero_grad : grad(b) = 0", ag_grad(t, b), 0.0f);

    ag_tape_free(t);
}

/* ════════════════════════════════════════════════════════
   MAIN
   ════════════════════════════════════════════════════════ */

int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║        test_autograd — suite complete       ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    /* ops scalaires */
    test_add();
    test_sub();
    test_mul();
    test_div();
    test_neg();
    test_pow();
    test_relu();
    test_tanh();
    test_sigmoid();
    test_exp();
    test_log();

    /* chain rule */
    test_chain_rule();
    test_shared_node();
    test_mse_loss();

    /* ops tensorielles */
    test_sum();
    test_matmul();
    test_transpose();

    /* memoire */
    test_tape_reset();
    test_zero_grad();

    summary();
    return _failed > 0 ? 1 : 0;
}
