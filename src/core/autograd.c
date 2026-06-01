#include "minitorch/core/autograd.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ════════════════════════════════════════════════════════
   STRUCTURES INTERNES
   ════════════════════════════════════════════════════════ */

typedef struct {
    float data;
    float grad;
} AgNode;

struct AgTape {
    AgNode  nodes  [AG_MAX_NODES];
    AgEntry entries[AG_MAX_NODES];
    int     n;
};

/* ════════════════════════════════════════════════════════
   CREATION / DESTRUCTION
   ════════════════════════════════════════════════════════ */

AgTape *ag_tape_create(void) {
    return (AgTape *)calloc(1, sizeof(AgTape));
}

void ag_tape_free(AgTape *t) {
    free(t);
}

void ag_reset(AgTape *t) {
    memset(t, 0, sizeof(*t));
}

void ag_zero_grad(AgTape *t) {
    for (int i = 0; i < t->n; i++)
        t->nodes[i].grad = 0.0f;
}

/* ════════════════════════════════════════════════════════
   ACCESSEURS
   ════════════════════════════════════════════════════════ */

float ag_data(const AgTape *t, AgVal v) { return t->nodes[v].data; }
float ag_grad(const AgTape *t, AgVal v) { return t->nodes[v].grad; }
void  ag_set_data(AgTape *t, AgVal v, float d) { t->nodes[v].data = d; }

/* ════════════════════════════════════════════════════════
   ALLOCATION DE NOEUDS
   ════════════════════════════════════════════════════════ */

static AgVal alloc_node(AgTape *t, float value) {
    assert(t->n < AG_MAX_NODES && "ag tape overflow");
    int idx = t->n++;
    t->nodes[idx].data = value;
    t->nodes[idx].grad = 0.0f;
    memset(&t->entries[idx], 0, sizeof(AgEntry));
    return idx;
}

AgVal ag_leaf(AgTape *t, float value) {
    return alloc_node(t, value);
}

AgVal ag_op(AgTape *t,
            AgBackwardFn fn,
            float result,
            const int   *inputs, int n_inputs,
            const float *saved,  int n_saved) {
    AgVal idx   = alloc_node(t, result);
    AgEntry *e  = &t->entries[idx];
    e->fn       = fn;
    e->n_inputs = n_inputs;
    e->n_saved  = n_saved;
    for (int i = 0; i < n_inputs; i++) e->inputs[i] = inputs[i];
    for (int i = 0; i < n_saved;  i++) e->saved[i]  = saved[i];
    return idx;
}

/* ════════════════════════════════════════════════════════
   BACKWARD
   ════════════════════════════════════════════════════════ */

void ag_backward(AgTape *t, AgVal loss) {
    static float grads[AG_MAX_NODES];
    memset(grads, 0, sizeof(float) * (size_t)t->n);
    grads[loss] = 1.0f;

    for (int i = loss; i >= 0; i--)
        if (t->entries[i].fn)
            t->entries[i].fn(&t->entries[i], grads, i);

    for (int i = 0; i < t->n; i++)
        t->nodes[i].grad += grads[i];
}

/* ════════════════════════════════════════════════════════
   OPS SCALAIRES
   ════════════════════════════════════════════════════════ */

/* ADD */
static void _bwd_add(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o];
    g[e->inputs[1]] += g[o];
}
AgVal ag_add(AgTape *t, AgVal a, AgVal b) {
    int ins[] = {a, b};
    return ag_op(t, _bwd_add, t->nodes[a].data + t->nodes[b].data,
                 ins, 2, NULL, 0);
}

/* SUB */
static void _bwd_sub(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] +=  g[o];
    g[e->inputs[1]] += -g[o];
}
AgVal ag_sub(AgTape *t, AgVal a, AgVal b) {
    int ins[] = {a, b};
    return ag_op(t, _bwd_sub, t->nodes[a].data - t->nodes[b].data,
                 ins, 2, NULL, 0);
}

/* MUL */
static void _bwd_mul(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] * e->saved[1];
    g[e->inputs[1]] += g[o] * e->saved[0];
}
AgVal ag_mul(AgTape *t, AgVal a, AgVal b) {
    float da = t->nodes[a].data, db = t->nodes[b].data;
    float sv[] = {da, db};
    int   ins[] = {a, b};
    return ag_op(t, _bwd_mul, da * db, ins, 2, sv, 2);
}

/* DIV */
static void _bwd_div(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] / e->saved[1];
    g[e->inputs[1]] -= g[o] * e->saved[0] / (e->saved[1] * e->saved[1]);
}
AgVal ag_div(AgTape *t, AgVal a, AgVal b) {
    float da = t->nodes[a].data, db = t->nodes[b].data;
    float sv[] = {da, db};
    int   ins[] = {a, b};
    return ag_op(t, _bwd_div, da / db, ins, 2, sv, 2);
}

/* NEG */
static void _bwd_neg(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] -= g[o];
}
AgVal ag_neg(AgTape *t, AgVal a) {
    int ins[] = {a};
    return ag_op(t, _bwd_neg, -t->nodes[a].data, ins, 1, NULL, 0);
}

/* POW */
static void _bwd_pow(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] * e->saved[1] * powf(e->saved[0], e->saved[1] - 1.0f);
}
AgVal ag_pow(AgTape *t, AgVal a, float exp) {
    float da = t->nodes[a].data;
    float sv[] = {da, exp};
    int   ins[] = {a};
    return ag_op(t, _bwd_pow, powf(da, exp), ins, 1, sv, 2);
}

/* RELU */
static void _bwd_relu(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] * (e->saved[0] > 0.0f ? 1.0f : 0.0f);
}
AgVal ag_relu(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    return ag_op(t, _bwd_relu, da > 0.0f ? da : 0.0f, ins, 1, sv, 1);
}

/* TANH */
static void _bwd_tanh(const AgEntry *e, float *g, int o) {
    float z = e->saved[0];
    g[e->inputs[0]] += g[o] * (1.0f - z * z);
}
AgVal ag_tanh(AgTape *t, AgVal a) {
    float z  = tanhf(t->nodes[a].data);
    float sv[] = {z};
    int   ins[] = {a};
    return ag_op(t, _bwd_tanh, z, ins, 1, sv, 1);
}

/* SIGMOID */
static void _bwd_sigmoid(const AgEntry *e, float *g, int o) {
    float s = e->saved[0];
    g[e->inputs[0]] += g[o] * s * (1.0f - s);
}
AgVal ag_sigmoid(AgTape *t, AgVal a) {
    float s  = 1.0f / (1.0f + expf(-t->nodes[a].data));
    float sv[] = {s};
    int   ins[] = {a};
    return ag_op(t, _bwd_sigmoid, s, ins, 1, sv, 1);
}

/* EXP */
static void _bwd_exp(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] * e->saved[0];
}
AgVal ag_exp(AgTape *t, AgVal a) {
    float z  = expf(t->nodes[a].data);
    float sv[] = {z};
    int   ins[] = {a};
    return ag_op(t, _bwd_exp, z, ins, 1, sv, 1);
}

/* LOG */
static void _bwd_log(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] / e->saved[0];
}
AgVal ag_log(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    return ag_op(t, _bwd_log, logf(da), ins, 1, sv, 1);
}

/* ════════════════════════════════════════════════════════
   OPS TENSORIELLES
   ════════════════════════════════════════════════════════ */

/*
 * SUM
 * z = vals[0] + vals[1] + ... + vals[n-1]
 * dz/d(vals[i]) = 1  pour tout i
 *
 * Le tape ne supporte que AG_MAX_INPUTS parents par noeud.
 * Pour n > AG_MAX_INPUTS on chaine des additions deux a deux.
 */
AgVal ag_sum(AgTape *t, const AgVal *vals, int n) {
    assert(n >= 1);
    AgVal acc = vals[0];
    for (int i = 1; i < n; i++)
        acc = ag_add(t, acc, vals[i]);
    return acc;
}

/*
 * MATMUL  A(m x k) @ B(k x n) = C(m x n)
 * C[i,j] = sum_l A[i,l] * B[l,j]
 * dL/dA[i,l] += dL/dC[i,j] * B[l,j]   (accumule par backward de mul+sum)
 * dL/dB[l,j] += dL/dC[i,j] * A[i,l]   (idem)
 */
void ag_matmul(AgTape *t,
               const AgVal *A, int m, int k,
               const AgVal *B, int n,
               AgVal *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            /* produits partiels pour cette case */
            AgVal terms[AG_MAX_NODES];  /* k termes au max */
            assert(k <= AG_MAX_NODES);
            for (int l = 0; l < k; l++)
                terms[l] = ag_mul(t, A[i * k + l], B[l * n + j]);
            C[i * n + j] = ag_sum(t, terms, k);
        }
    }
}

/*
 * TRANSPOSE  A(rows x cols) -> At(cols x rows)
 * At[j, i] = A[i, j]
 * Pas de noeud propre : on recopie juste les handles.
 * Le backward est gratuit car ce sont les memes AgVal.
 */
void ag_transpose(AgTape *t,
                  const AgVal *A, int rows, int cols,
                  AgVal *At) {
    (void)t;  /* pas d'op sur le tape, simple reindexation */
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            At[j * rows + i] = A[i * cols + j];
}
