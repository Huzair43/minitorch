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

static void visit_backward_order(const AgTape *t,
                                 int node,
                                 unsigned char *seen,
                                 int *order,
                                 int *count) {
    if (node < 0 || node >= t->n || seen[node]) {
        return;
    }

    seen[node] = 1;

    const AgEntry *entry = &t->entries[node];
    for (int i = 0; i < entry->n_inputs; i++) {
        visit_backward_order(t, entry->inputs[i], seen, order, count);
    }

    order[(*count)++] = node;
}

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
    if (!t) return;
    memset(t, 0, sizeof(*t));
}

void ag_zero_grad(AgTape *t) {
    if (!t) return;
    for (int i = 0; i < t->n; i++)
        t->nodes[i].grad = 0.0f;
}

int ag_checkpoint(const AgTape *t) {
    if (!t) return 0;
    return t->n;
}

void ag_rewind(AgTape *t, int checkpoint) {
    if (!t || checkpoint < 0 || checkpoint > t->n) return;

    for (int i = checkpoint; i < t->n; i++) {
        memset(&t->nodes[i], 0, sizeof(AgNode));
        memset(&t->entries[i], 0, sizeof(AgEntry));
    }
    t->n = checkpoint;
}

int ag_node_count(const AgTape *t) {
    if (!t) return 0;
    return t->n;
}

/* ════════════════════════════════════════════════════════
   ACCESSEURS
   ════════════════════════════════════════════════════════ */

float ag_data(const AgTape *t, AgVal v) {
    if (!t || v < 0 || v >= t->n) return NAN;
    return t->nodes[v].data;
}

float ag_grad(const AgTape *t, AgVal v) {
    if (!t || v < 0 || v >= t->n) return NAN;
    return t->nodes[v].grad;
}

void  ag_set_data(AgTape *t, AgVal v, float d) {
    if (!t || v < 0 || v >= t->n) return;
    t->nodes[v].data = d;
}

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
    assert(t);
    assert(n_inputs >= 0 && n_inputs <= AG_MAX_INPUTS);
    assert(n_saved >= 0 && n_saved <= AG_MAX_SAVED);
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
    if (!t || loss < 0 || loss >= t->n) return;

    float *grads = (float *)calloc((size_t)t->n, sizeof(float));
    unsigned char *seen = (unsigned char *)calloc((size_t)t->n, sizeof(unsigned char));
    int *order = (int *)malloc((size_t)t->n * sizeof(int));

    if (!grads || !seen || !order) {
        free(grads);
        free(seen);
        free(order);
        return;
    }

    int count = 0;
    visit_backward_order(t, loss, seen, order, &count);
    grads[loss] = 1.0f;

    for (int i = count - 1; i >= 0; i--) {
        int node = order[i];
        if (t->entries[node].fn) {
            t->entries[node].fn(&t->entries[node], grads, node);
        }
    }

    for (int i = 0; i < t->n; i++) {
        t->nodes[i].grad += grads[i];
    }

    free(grads);
    free(seen);
    free(order);
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
    float denom = e->saved[1];
    if (denom != 0.0f) {
        g[e->inputs[0]] += g[o] / denom;
        g[e->inputs[1]] -= g[o] * e->saved[0] / (denom * denom);
    } else {
        g[e->inputs[0]] += NAN;
        g[e->inputs[1]] += NAN;
    }
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

/* LEAKY RELU */
static void _bwd_leaky_relu(const AgEntry *e, float *g, int o) {
    float slope = e->saved[1];
    g[e->inputs[0]] += g[o] * (e->saved[0] > 0.0f ? 1.0f : slope);
}

AgVal ag_leaky_relu(AgTape *t, AgVal a, float negative_slope) {
    float da = t->nodes[a].data;
    float sv[] = {da, negative_slope};
    int   ins[] = {a};
    float value = da > 0.0f ? da : negative_slope * da;
    return ag_op(t, _bwd_leaky_relu, value, ins, 1, sv, 2);
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
    float x = e->saved[0];
    if (x > 0.0f) {
        g[e->inputs[0]] += g[o] / x;
    } else {
        g[e->inputs[0]] += NAN;
    }
}
AgVal ag_log(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    float value = da > 0.0f ? logf(da) : NAN;
    return ag_op(t, _bwd_log, value, ins, 1, sv, 1);
}

/* SQRT */
static void _bwd_sqrt(const AgEntry *e, float *g, int o) {
    float x = e->saved[0];
    if (x > 0.0f) {
        g[e->inputs[0]] += g[o] * (0.5f / sqrtf(x));
    }
}

AgVal ag_sqrt(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    float value = da >= 0.0f ? sqrtf(da) : NAN;
    return ag_op(t, _bwd_sqrt, value, ins, 1, sv, 1);
}

/* ABS */
static void _bwd_abs(const AgEntry *e, float *g, int o) {
    float x = e->saved[0];
    float sign = (x > 0.0f) ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    g[e->inputs[0]] += g[o] * sign;
}

AgVal ag_abs(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    return ag_op(t, _bwd_abs, fabsf(da), ins, 1, sv, 1);
}

/* SQUARE */
static void _bwd_square(const AgEntry *e, float *g, int o) {
    g[e->inputs[0]] += g[o] * (2.0f * e->saved[0]);
}

AgVal ag_square(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    return ag_op(t, _bwd_square, da * da, ins, 1, sv, 1);
}

/* SOFTPLUS */
static void _bwd_softplus(const AgEntry *e, float *g, int o) {
    float x = e->saved[0];
    float sigmoid = 1.0f / (1.0f + expf(-x));
    g[e->inputs[0]] += g[o] * sigmoid;
}

AgVal ag_softplus(AgTape *t, AgVal a) {
    float da = t->nodes[a].data;
    float sv[] = {da};
    int   ins[] = {a};
    float value = fmaxf(da, 0.0f) + log1pf(expf(-fabsf(da)));
    return ag_op(t, _bwd_softplus, value, ins, 1, sv, 1);
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
            AgVal *terms = (AgVal *)malloc(sizeof(AgVal) * (size_t)k);
            if (!terms) return;
            for (int l = 0; l < k; l++)
                terms[l] = ag_mul(t, A[i * k + l], B[l * n + j]);
            C[i * n + j] = ag_sum(t, terms, k);
            free(terms);
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

/* ════════════════════════════════════════════════════════
   HELPERS TENSORIELS
   ════════════════════════════════════════════════════════ */

void ag_tensor_copy(AgVal *dst, const AgVal *src, int n) {
    if (!dst || !src || n < 0) return;
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void ag_tensor_reshape_view(AgVal *dst, const AgVal *src, int n) {
    ag_tensor_copy(dst, src, n);
}

void ag_tensor_apply_unary(AgTape *t, const AgVal *src, int n, AgVal *dst, AgUnaryFn fn) {
    if (!t || !src || !dst || !fn || n < 0) return;
    for (int i = 0; i < n; i++) {
        dst[i] = fn(t, src[i]);
    }
}

void ag_tensor_apply_binary(AgTape *t, const AgVal *lhs, const AgVal *rhs, int n, AgVal *dst, AgBinaryFn fn) {
    if (!t || !lhs || !rhs || !dst || !fn || n < 0) return;
    for (int i = 0; i < n; i++) {
        dst[i] = fn(t, lhs[i], rhs[i]);
    }
}

void ag_tensor_broadcast_binary_2d(AgTape *t,
                                   const AgVal *lhs, int lhs_rows, int lhs_cols,
                                   const AgVal *rhs, int rhs_rows, int rhs_cols,
                                   AgVal *out, AgBinaryFn fn) {
    if (!t || !lhs || !rhs || !out || !fn) return;
    if (lhs_rows <= 0 || lhs_cols <= 0 || rhs_rows <= 0 || rhs_cols <= 0) return;

    int out_rows = lhs_rows > rhs_rows ? lhs_rows : rhs_rows;
    int out_cols = lhs_cols > rhs_cols ? lhs_cols : rhs_cols;

    if ((lhs_rows != out_rows && lhs_rows != 1) ||
        (rhs_rows != out_rows && rhs_rows != 1) ||
        (lhs_cols != out_cols && lhs_cols != 1) ||
        (rhs_cols != out_cols && rhs_cols != 1)) {
        return;
    }

    for (int i = 0; i < out_rows; i++) {
        int lhs_row = lhs_rows == 1 ? 0 : i;
        int rhs_row = rhs_rows == 1 ? 0 : i;
        for (int j = 0; j < out_cols; j++) {
            int lhs_col = lhs_cols == 1 ? 0 : j;
            int rhs_col = rhs_cols == 1 ? 0 : j;
            out[i * out_cols + j] = fn(t, lhs[lhs_row * lhs_cols + lhs_col], rhs[rhs_row * rhs_cols + rhs_col]);
        }
    }
}

void ag_tensor_reduce_sum_2d(AgTape *t,
                             const AgVal *src, int rows, int cols,
                             int axis, AgVal *out) {
    if (!t || !src || !out) return;
    if (rows <= 0 || cols <= 0) return;
    if (axis != 0 && axis != 1) return;

    if (axis == 0) {
        for (int j = 0; j < cols; j++) {
            AgVal *terms = (AgVal *)malloc(sizeof(AgVal) * (size_t)rows);
            if (!terms) return;
            for (int i = 0; i < rows; i++) {
                terms[i] = src[i * cols + j];
            }
            out[j] = ag_sum(t, terms, rows);
            free(terms);
        }
    } else {
        for (int i = 0; i < rows; i++) {
            AgVal *terms = (AgVal *)malloc(sizeof(AgVal) * (size_t)cols);
            if (!terms) return;
            for (int j = 0; j < cols; j++) {
                terms[j] = src[i * cols + j];
            }
            out[i] = ag_sum(t, terms, cols);
            free(terms);
        }
    }
}

void ag_tensor_reduce_mean_2d(AgTape *t,
                              const AgVal *src, int rows, int cols,
                              int axis, AgVal *out) {
    if (!t || !src || !out) return;
    if (rows <= 0 || cols <= 0) return;
    if (axis != 0 && axis != 1) return;

    if (axis == 0) {
        AgVal inv = ag_leaf(t, 1.0f / (float)rows);
        for (int j = 0; j < cols; j++) {
            AgVal *terms = (AgVal *)malloc(sizeof(AgVal) * (size_t)rows);
            if (!terms) return;
            for (int i = 0; i < rows; i++) {
                terms[i] = src[i * cols + j];
            }
            out[j] = ag_mul(t, ag_sum(t, terms, rows), inv);
            free(terms);
        }
    } else {
        AgVal inv = ag_leaf(t, 1.0f / (float)cols);
        for (int i = 0; i < rows; i++) {
            AgVal *terms = (AgVal *)malloc(sizeof(AgVal) * (size_t)cols);
            if (!terms) return;
            for (int j = 0; j < cols; j++) {
                terms[j] = src[i * cols + j];
            }
            out[i] = ag_mul(t, ag_sum(t, terms, cols), inv);
            free(terms);
        }
    }
}
