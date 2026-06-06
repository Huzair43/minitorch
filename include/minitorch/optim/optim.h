#ifndef MINITORCH_OPTIM_H
#define MINITORCH_OPTIM_H

#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"

typedef enum {
    MT_OPT_SGD = 0,
    MT_OPT_MOMENTUM,
    MT_OPT_ADAM
} MtOptimKind;

typedef struct {
    MtOptimKind kind;
    float lr;
    float momentum;
    float beta1;
    float beta2;
    float eps;
    int step;
    int n_params;
    int capacity;
    AgVal *params;
    float *velocity;
    float *m;
    float *v;
} MtOptimizer;

MtOptimizer *mt_sgd_create(float lr);
MtOptimizer *mt_momentum_create(float lr, float momentum);
MtOptimizer *mt_adam_create(float lr, float beta1, float beta2, float eps);
void         mt_optimizer_free(MtOptimizer *opt);

int          mt_optimizer_add_param(MtOptimizer *opt, AgVal param);
int          mt_optimizer_add_params(MtOptimizer *opt, const AgVal *params, int n);
int          mt_optimizer_add_linear(MtOptimizer *opt, const MtLinear *layer);
int          mt_optimizer_add_sequential(MtOptimizer *opt, const MtSequential *seq);
int          mt_optimizer_add_model(MtOptimizer *opt, const MtModel *model);

void         mt_optimizer_zero_grad(AgTape *t, MtOptimizer *opt);
void         mt_optimizer_step(AgTape *t, MtOptimizer *opt);

#endif /* MINITORCH_OPTIM_H */
