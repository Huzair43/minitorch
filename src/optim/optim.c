#include "minitorch/optim/optim.h"

#include <math.h>
#include <stdlib.h>

static MtOptimizer *mt_optimizer_create(MtOptimKind kind, float lr) {
    if (lr <= 0.0f) {
        return NULL;
    }

    MtOptimizer *opt = (MtOptimizer *)calloc(1, sizeof(MtOptimizer));
    if (!opt) {
        return NULL;
    }

    opt->kind = kind;
    opt->lr = lr;
    opt->momentum = 0.0f;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    return opt;
}

static int mt_optimizer_reserve(MtOptimizer *opt, int needed) {
    if (!opt || needed <= opt->capacity) {
        return opt != NULL;
    }

    int new_capacity = opt->capacity == 0 ? 8 : opt->capacity;
    while (new_capacity < needed) {
        new_capacity *= 2;
    }

    AgVal *new_params = (AgVal *)realloc(opt->params, sizeof(AgVal) * (size_t)new_capacity);
    if (!new_params) {
        return 0;
    }
    opt->params = new_params;

    float *new_velocity = (float *)realloc(opt->velocity, sizeof(float) * (size_t)new_capacity);
    if (!new_velocity) {
        return 0;
    }
    opt->velocity = new_velocity;

    float *new_m = (float *)realloc(opt->m, sizeof(float) * (size_t)new_capacity);
    if (!new_m) {
        return 0;
    }
    opt->m = new_m;

    float *new_v = (float *)realloc(opt->v, sizeof(float) * (size_t)new_capacity);
    if (!new_v) {
        return 0;
    }
    opt->v = new_v;

    for (int i = opt->capacity; i < new_capacity; i++) {
        opt->velocity[i] = 0.0f;
        opt->m[i] = 0.0f;
        opt->v[i] = 0.0f;
    }

    opt->capacity = new_capacity;
    return 1;
}

MtOptimizer *mt_sgd_create(float lr) {
    return mt_optimizer_create(MT_OPT_SGD, lr);
}

MtOptimizer *mt_momentum_create(float lr, float momentum) {
    if (momentum < 0.0f) {
        return NULL;
    }

    MtOptimizer *opt = mt_optimizer_create(MT_OPT_MOMENTUM, lr);
    if (!opt) {
        return NULL;
    }
    opt->momentum = momentum;
    return opt;
}

MtOptimizer *mt_adam_create(float lr, float beta1, float beta2, float eps) {
    if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || eps <= 0.0f) {
        return NULL;
    }

    MtOptimizer *opt = mt_optimizer_create(MT_OPT_ADAM, lr);
    if (!opt) {
        return NULL;
    }
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    return opt;
}

void mt_optimizer_free(MtOptimizer *opt) {
    if (!opt) {
        return;
    }
    free(opt->params);
    free(opt->velocity);
    free(opt->m);
    free(opt->v);
    free(opt);
}

int mt_optimizer_add_param(MtOptimizer *opt, AgVal param) {
    if (!opt || param < 0) {
        return 0;
    }
    if (!mt_optimizer_reserve(opt, opt->n_params + 1)) {
        return 0;
    }

    opt->params[opt->n_params] = param;
    opt->velocity[opt->n_params] = 0.0f;
    opt->m[opt->n_params] = 0.0f;
    opt->v[opt->n_params] = 0.0f;
    opt->n_params++;
    return 1;
}

int mt_optimizer_add_params(MtOptimizer *opt, const AgVal *params, int n) {
    if (!opt || !params || n < 0) {
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (!mt_optimizer_add_param(opt, params[i])) {
            return 0;
        }
    }
    return 1;
}

int mt_optimizer_add_linear(MtOptimizer *opt, const MtLinear *layer) {
    if (!opt || !layer || !layer->weight) {
        return 0;
    }

    int n_weights = layer->in_features * layer->out_features;
    if (!mt_optimizer_add_params(opt, layer->weight, n_weights)) {
        return 0;
    }
    if (layer->use_bias && layer->bias) {
        return mt_optimizer_add_params(opt, layer->bias, layer->out_features);
    }
    return 1;
}

int mt_optimizer_add_sequential(MtOptimizer *opt, const MtSequential *seq) {
    if (!opt || !seq) {
        return 0;
    }
    for (int i = 0; i < seq->n_layers; i++) {
        if (seq->layers[i].kind == MT_LAYER_LINEAR) {
            if (!mt_optimizer_add_linear(opt, seq->layers[i].linear)) {
                return 0;
            }
        }
    }
    return 1;
}

int mt_optimizer_add_model(MtOptimizer *opt, const MtModel *model) {
    if (!opt || !model) {
        return 0;
    }

    if (model->kind == MT_MODEL_LINEAR_BINARY || model->kind == MT_MODEL_LINEAR_MULTICLASS) {
        return mt_optimizer_add_linear(opt, model->linear);
    }
    if (model->kind == MT_MODEL_MLP_BINARY || model->kind == MT_MODEL_MLP_MULTICLASS) {
        return mt_optimizer_add_sequential(opt, model->seq);
    }
    return 0;
}

void mt_optimizer_zero_grad(AgTape *t, MtOptimizer *opt) {
    (void)opt;
    ag_zero_grad(t);
}

void mt_optimizer_step(AgTape *t, MtOptimizer *opt) {
    if (!t || !opt) {
        return;
    }

    opt->step++;
    for (int i = 0; i < opt->n_params; i++) {
        AgVal param = opt->params[i];
        float grad = ag_grad(t, param);
        float value = ag_data(t, param);
        float update = 0.0f;

        switch (opt->kind) {
            case MT_OPT_SGD:
                update = grad;
                break;
            case MT_OPT_MOMENTUM:
                opt->velocity[i] = opt->momentum * opt->velocity[i] + grad;
                update = opt->velocity[i];
                break;
            case MT_OPT_ADAM: {
                opt->m[i] = opt->beta1 * opt->m[i] + (1.0f - opt->beta1) * grad;
                opt->v[i] = opt->beta2 * opt->v[i] + (1.0f - opt->beta2) * grad * grad;
                float m_hat = opt->m[i] / (1.0f - powf(opt->beta1, (float)opt->step));
                float v_hat = opt->v[i] / (1.0f - powf(opt->beta2, (float)opt->step));
                update = m_hat / (sqrtf(v_hat) + opt->eps);
                break;
            }
            default:
                update = grad;
                break;
        }

        ag_set_data(t, param, value - opt->lr * update);
    }
}
