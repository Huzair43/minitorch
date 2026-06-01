#ifndef MINITORCH_AUTOGRAD_H
#define MINITORCH_AUTOGRAD_H

#include <stddef.h>

/* ── Limites configurables ─────────────────────────────── */
#ifndef AG_MAX_NODES
#  define AG_MAX_NODES  65536
#endif
#ifndef AG_MAX_INPUTS
#  define AG_MAX_INPUTS 4
#endif
#ifndef AG_MAX_SAVED
#  define AG_MAX_SAVED  8
#endif

/* ── Types ─────────────────────────────────────────────── */
typedef struct AgTape   AgTape;
typedef int             AgVal;

typedef struct AgEntry AgEntry;
typedef void (*AgBackwardFn)(const AgEntry *e,
                              float        *grads,
                              int           out_idx);

struct AgEntry {
    AgBackwardFn fn;
    int          inputs[AG_MAX_INPUTS];
    int          n_inputs;
    float        saved [AG_MAX_SAVED];
    int          n_saved;
};

/* ── API principale ────────────────────────────────────── */
AgTape *ag_tape_create(void);
void    ag_tape_free(AgTape *t);
void    ag_reset(AgTape *t);
void    ag_zero_grad(AgTape *t);

AgVal   ag_leaf(AgTape *t, float value);
AgVal   ag_op(AgTape *t, AgBackwardFn fn, float result,
              const int *inputs, int n_inputs,
              const float *saved, int n_saved);

void    ag_backward(AgTape *t, AgVal loss);

float   ag_data(const AgTape *t, AgVal v);
float   ag_grad(const AgTape *t, AgVal v);
void    ag_set_data(AgTape *t, AgVal v, float data);

/* ── Ops scalaires ─────────────────────────────────────── */
AgVal   ag_add(AgTape *t, AgVal a, AgVal b);
AgVal   ag_sub(AgTape *t, AgVal a, AgVal b);
AgVal   ag_mul(AgTape *t, AgVal a, AgVal b);
AgVal   ag_div(AgTape *t, AgVal a, AgVal b);
AgVal   ag_neg(AgTape *t, AgVal a);
AgVal   ag_pow(AgTape *t, AgVal a, float exp);

/* ── Activations ───────────────────────────────────────── */
AgVal   ag_relu(AgTape *t, AgVal a);
AgVal   ag_tanh(AgTape *t, AgVal a);
AgVal   ag_sigmoid(AgTape *t, AgVal a);
AgVal   ag_exp(AgTape *t, AgVal a);
AgVal   ag_log(AgTape *t, AgVal a);

/* ── Ops tensorielles (sur tableaux de AgVal) ──────────── */

/*
 * ag_sum : somme d'un tableau de n AgVal
 * retourne un seul AgVal = vals[0] + ... + vals[n-1]
 */
AgVal   ag_sum(AgTape *t, const AgVal *vals, int n);

/*
 * ag_matmul : produit matriciel A (m x k) @ B (k x n)
 *   A, B  : tableaux row-major de AgVal (taille m*k et k*n)
 *   C     : tableau de sortie pre-alloue (taille m*n)
 * Remplit C[i*n+j] = sum_l A[i*k+l] * B[l*n+j]
 */
void    ag_matmul(AgTape *t,
                  const AgVal *A, int m, int k,
                  const AgVal *B, int n,
                  AgVal *C);

/*
 * ag_transpose : transpose A (rows x cols)
 *   At : tableau de sortie pre-alloue (taille rows*cols)
 * Remplit At[j*rows+i] = A[i*cols+j]
 */
void    ag_transpose(AgTape *t,
                     const AgVal *A, int rows, int cols,
                     AgVal *At);

#endif /* MINITORCH_AUTOGRAD_H */
