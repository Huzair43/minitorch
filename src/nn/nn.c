#include "minitorch/nn/nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float mt_randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static AgVal mt_zero(AgTape *t) {
    return ag_leaf(t, 0.0f);
}

MtLinear *mt_linear_create(AgTape *t, int in_features, int out_features, int use_bias) {
    if (!t || in_features <= 0 || out_features <= 0) {
        return NULL;
    }

    MtLinear *layer = (MtLinear *)calloc(1, sizeof(MtLinear));
    if (!layer) {
        return NULL;
    }

    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->use_bias = use_bias ? 1 : 0;
    layer->weight = (AgVal *)malloc(sizeof(AgVal) * (size_t)(in_features * out_features));
    if (!layer->weight) {
        free(layer);
        return NULL;
    }

    layer->bias = NULL;
    if (layer->use_bias) {
        layer->bias = (AgVal *)malloc(sizeof(AgVal) * (size_t)out_features);
        if (!layer->bias) {
            free(layer->weight);
            free(layer);
            return NULL;
        }
    }

    for (int i = 0; i < in_features * out_features; i++) {
        layer->weight[i] = mt_zero(t);
    }
    for (int i = 0; i < out_features; i++) {
        if (layer->use_bias) {
            layer->bias[i] = mt_zero(t);
        }
    }

    return layer;
}

void mt_linear_free(MtLinear *layer) {
    if (!layer) {
        return;
    }
    free(layer->weight);
    free(layer->bias);
    free(layer);
}

void mt_linear_init_uniform(AgTape *t, MtLinear *layer, float lo, float hi) {
    if (!t || !layer) {
        return;
    }
    for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
        for (int in_idx = 0; in_idx < layer->in_features; in_idx++) {
            mt_linear_set_weight(t, layer, out_idx, in_idx, mt_randf(lo, hi));
        }
        if (layer->use_bias) {
            mt_linear_set_bias(t, layer, out_idx, 0.0f);
        }
    }
}

void mt_linear_init_zeros(AgTape *t, MtLinear *layer) {
    mt_linear_init_uniform(t, layer, 0.0f, 0.0f);
}

void mt_linear_init_xavier_uniform(AgTape *t, MtLinear *layer) {
    if (!t || !layer || layer->in_features <= 0 || layer->out_features <= 0) {
        return;
    }

    float limit = sqrtf(6.0f / (float)(layer->in_features + layer->out_features));
    mt_linear_init_uniform(t, layer, -limit, limit);
}

void mt_linear_init_he_uniform(AgTape *t, MtLinear *layer) {
    if (!t || !layer || layer->in_features <= 0) {
        return;
    }

    float limit = sqrtf(6.0f / (float)layer->in_features);
    mt_linear_init_uniform(t, layer, -limit, limit);
}

void mt_linear_set_weight(AgTape *t, MtLinear *layer, int out_idx, int in_idx, float value) {
    if (!t || !layer || !layer->weight) {
        return;
    }
    if (out_idx < 0 || out_idx >= layer->out_features || in_idx < 0 || in_idx >= layer->in_features) {
        return;
    }
    ag_set_data(t, layer->weight[out_idx * layer->in_features + in_idx], value);
}

void mt_linear_set_bias(AgTape *t, MtLinear *layer, int out_idx, float value) {
    if (!t || !layer || !layer->use_bias || !layer->bias) {
        return;
    }
    if (out_idx < 0 || out_idx >= layer->out_features) {
        return;
    }
    ag_set_data(t, layer->bias[out_idx], value);
}

AgVal mt_linear_weight(const MtLinear *layer, int out_idx, int in_idx) {
    if (!layer || !layer->weight) {
        return -1;
    }
    if (out_idx < 0 || out_idx >= layer->out_features || in_idx < 0 || in_idx >= layer->in_features) {
        return -1;
    }
    return layer->weight[out_idx * layer->in_features + in_idx];
}

AgVal mt_linear_bias(const MtLinear *layer, int out_idx) {
    if (!layer || !layer->use_bias || !layer->bias) {
        return -1;
    }
    if (out_idx < 0 || out_idx >= layer->out_features) {
        return -1;
    }
    return layer->bias[out_idx];
}

void mt_linear_forward(AgTape *t, const MtLinear *layer, const AgVal *input, AgVal *output) {
    if (!t || !layer || !input || !output || !layer->weight) {
        return;
    }

    for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
        AgVal acc = layer->use_bias ? layer->bias[out_idx] : mt_zero(t);
        for (int in_idx = 0; in_idx < layer->in_features; in_idx++) {
            AgVal w = layer->weight[out_idx * layer->in_features + in_idx];
            acc = ag_add(t, acc, ag_mul(t, input[in_idx], w));
        }
        output[out_idx] = acc;
    }
}

void mt_linear_step(AgTape *t, MtLinear *layer, float lr) {
    if (!t || !layer || !layer->weight) {
        return;
    }

    for (int i = 0; i < layer->in_features * layer->out_features; i++) {
        AgVal param = layer->weight[i];
        ag_set_data(t, param, ag_data(t, param) - lr * ag_grad(t, param));
    }

    if (layer->use_bias && layer->bias) {
        for (int i = 0; i < layer->out_features; i++) {
            AgVal param = layer->bias[i];
            ag_set_data(t, param, ag_data(t, param) - lr * ag_grad(t, param));
        }
    }
}

void mt_activation_forward(AgTape *t, MtActivation activation, const AgVal *input, int n, AgVal *output) {
    if (!t || !input || !output || n < 0) {
        return;
    }

    for (int i = 0; i < n; i++) {
        switch (activation) {
            case MT_ACT_RELU:
                output[i] = ag_relu(t, input[i]);
                break;
            case MT_ACT_SIGMOID:
                output[i] = ag_sigmoid(t, input[i]);
                break;
            case MT_ACT_TANH:
                output[i] = ag_tanh(t, input[i]);
                break;
            default:
                output[i] = input[i];
                break;
        }
    }
}

void mt_relu(AgTape *t, const AgVal *input, int n, AgVal *output) {
    mt_activation_forward(t, MT_ACT_RELU, input, n, output);
}

void mt_sigmoid(AgTape *t, const AgVal *input, int n, AgVal *output) {
    mt_activation_forward(t, MT_ACT_SIGMOID, input, n, output);
}

void mt_tanh(AgTape *t, const AgVal *input, int n, AgVal *output) {
    mt_activation_forward(t, MT_ACT_TANH, input, n, output);
}

void mt_softmax(AgTape *t, const AgVal *logits, int n, AgVal *probs) {
    if (!t || !logits || !probs || n <= 0) {
        return;
    }

    AgVal *exp_vals = (AgVal *)malloc(sizeof(AgVal) * (size_t)n);
    if (!exp_vals) {
        return;
    }

    for (int i = 0; i < n; i++) {
        exp_vals[i] = ag_exp(t, logits[i]);
    }

    AgVal denom = ag_sum(t, exp_vals, n);
    for (int i = 0; i < n; i++) {
        probs[i] = ag_div(t, exp_vals[i], denom);
    }

    free(exp_vals);
}

MtSequential *mt_sequential_create(int max_width) {
    if (max_width <= 0) {
        return NULL;
    }

    MtSequential *seq = (MtSequential *)calloc(1, sizeof(MtSequential));
    if (!seq) {
        return NULL;
    }

    seq->max_width = max_width;
    seq->buffer_a = (AgVal *)malloc(sizeof(AgVal) * (size_t)max_width);
    seq->buffer_b = (AgVal *)malloc(sizeof(AgVal) * (size_t)max_width);
    if (!seq->buffer_a || !seq->buffer_b) {
        mt_sequential_free(seq);
        return NULL;
    }

    return seq;
}

void mt_sequential_free(MtSequential *seq) {
    if (!seq) {
        return;
    }
    free(seq->buffer_a);
    free(seq->buffer_b);
    free(seq);
}

int mt_sequential_add_linear(MtSequential *seq, MtLinear *layer) {
    if (!seq || !layer || seq->n_layers >= MT_MAX_SEQ_LAYERS) {
        return 0;
    }
    seq->layers[seq->n_layers].kind = MT_LAYER_LINEAR;
    seq->layers[seq->n_layers].linear = layer;
    seq->n_layers++;
    return 1;
}

int mt_sequential_add_activation(MtSequential *seq, MtActivation activation) {
    if (!seq || seq->n_layers >= MT_MAX_SEQ_LAYERS) {
        return 0;
    }
    seq->layers[seq->n_layers].kind = MT_LAYER_ACTIVATION;
    seq->layers[seq->n_layers].activation = activation;
    seq->layers[seq->n_layers].linear = NULL;
    seq->n_layers++;
    return 1;
}

int mt_sequential_forward(AgTape *t,
                          const MtSequential *seq,
                          const AgVal *input,
                          int input_size,
                          AgVal *output,
                          int output_size) {
    if (!t || !seq || !input || !output || input_size <= 0 || output_size <= 0) {
        return 0;
    }
    if (input_size > seq->max_width || output_size > seq->max_width) {
        return 0;
    }

    AgVal *current = seq->buffer_a;
    AgVal *next = seq->buffer_b;
    int current_size = input_size;

    for (int i = 0; i < input_size; i++) {
        current[i] = input[i];
    }

    for (int layer_idx = 0; layer_idx < seq->n_layers; layer_idx++) {
        const MtLayer *layer = &seq->layers[layer_idx];
        int next_size = current_size;

        if (layer->kind == MT_LAYER_LINEAR) {
            if (!layer->linear || current_size != layer->linear->in_features) {
                return 0;
            }
            next_size = layer->linear->out_features;
            if (next_size > seq->max_width) {
                return 0;
            }
            mt_linear_forward(t, layer->linear, current, next);
        } else if (layer->kind == MT_LAYER_ACTIVATION) {
            mt_activation_forward(t, layer->activation, current, current_size, next);
        } else {
            return 0;
        }

        AgVal *tmp = current;
        current = next;
        next = tmp;
        current_size = next_size;
    }

    if (current_size != output_size) {
        return 0;
    }
    for (int i = 0; i < output_size; i++) {
        output[i] = current[i];
    }
    return 1;
}

void mt_sequential_step(AgTape *t, MtSequential *seq, float lr) {
    if (!t || !seq) {
        return;
    }
    for (int i = 0; i < seq->n_layers; i++) {
        if (seq->layers[i].kind == MT_LAYER_LINEAR) {
            mt_linear_step(t, seq->layers[i].linear, lr);
        }
    }
}

AgVal mt_mse_loss(AgTape *t, const AgVal *pred, const AgVal *target, int n) {
    if (!t || !pred || !target || n <= 0) {
        return -1;
    }

    AgVal acc = mt_zero(t);
    for (int i = 0; i < n; i++) {
        AgVal diff = ag_sub(t, pred[i], target[i]);
        acc = ag_add(t, acc, ag_square(t, diff));
    }
    return ag_mul(t, acc, ag_leaf(t, 1.0f / (float)n));
}

AgVal mt_bce_loss(AgTape *t, const AgVal *pred, const AgVal *target, int n) {
    if (!t || !pred || !target || n <= 0) {
        return -1;
    }

    AgVal acc = mt_zero(t);
    AgVal one = ag_leaf(t, 1.0f);
    AgVal eps = ag_leaf(t, 1e-7f);

    for (int i = 0; i < n; i++) {
        AgVal log_p = ag_log(t, ag_add(t, pred[i], eps));
        AgVal log_not_p = ag_log(t, ag_add(t, ag_sub(t, one, pred[i]), eps));
        AgVal left = ag_mul(t, target[i], log_p);
        AgVal right = ag_mul(t, ag_sub(t, one, target[i]), log_not_p);
        acc = ag_add(t, acc, ag_neg(t, ag_add(t, left, right)));
    }

    return ag_mul(t, acc, ag_leaf(t, 1.0f / (float)n));
}

AgVal mt_cross_entropy_loss(AgTape *t, const AgVal *probs, const AgVal *target, int n) {
    if (!t || !probs || !target || n <= 0) {
        return -1;
    }

    AgVal acc = mt_zero(t);
    AgVal eps = ag_leaf(t, 1e-7f);

    for (int i = 0; i < n; i++) {
        AgVal log_prob = ag_log(t, ag_add(t, probs[i], eps));
        acc = ag_add(t, acc, ag_mul(t, target[i], log_prob));
    }

    return ag_neg(t, acc);
}

AgVal mt_cross_entropy_from_logits(AgTape *t, const AgVal *logits, const AgVal *target, int n) {
    if (!t || !logits || !target || n <= 0) {
        return -1;
    }

    float max_logit = ag_data(t, logits[0]);
    for (int i = 1; i < n; i++) {
        float value = ag_data(t, logits[i]);
        if (value > max_logit) {
            max_logit = value;
        }
    }

    AgVal max_node = ag_leaf(t, max_logit);
    AgVal *shifted_exp = (AgVal *)malloc(sizeof(AgVal) * (size_t)n);
    if (!shifted_exp) {
        return -1;
    }

    for (int i = 0; i < n; i++) {
        shifted_exp[i] = ag_exp(t, ag_sub(t, logits[i], max_node));
    }

    AgVal log_sum_exp = ag_add(t, ag_log(t, ag_sum(t, shifted_exp, n)), max_node);
    AgVal target_logit = mt_zero(t);
    for (int i = 0; i < n; i++) {
        target_logit = ag_add(t, target_logit, ag_mul(t, target[i], logits[i]));
    }

    free(shifted_exp);
    return ag_sub(t, log_sum_exp, target_logit);
}

MtLoss mt_loss_create(MtLossKind kind) {
    MtLoss loss;
    loss.kind = kind;
    return loss;
}

AgVal mt_loss_forward(AgTape *t, const MtLoss *loss, const AgVal *pred, const AgVal *target, int n) {
    if (!t || !loss || !pred || !target || n <= 0) {
        return -1;
    }

    switch (loss->kind) {
        case MT_LOSS_MSE:
            return mt_mse_loss(t, pred, target, n);
        case MT_LOSS_BCE:
            return mt_bce_loss(t, pred, target, n);
        case MT_LOSS_CROSS_ENTROPY:
            return mt_cross_entropy_loss(t, pred, target, n);
        case MT_LOSS_CROSS_ENTROPY_FROM_LOGITS:
            return mt_cross_entropy_from_logits(t, pred, target, n);
        default:
            return -1;
    }
}

int mt_argmax_values(const float *values, int n) {
    if (!values || n <= 0) {
        return -1;
    }

    int best = 0;
    float best_value = values[0];
    for (int i = 1; i < n; i++) {
        if (values[i] > best_value) {
            best = i;
            best_value = values[i];
        }
    }
    return best;
}

int mt_argmax(AgTape *t, const AgVal *values, int n) {
    if (!t || !values || n <= 0) {
        return -1;
    }

    int best = 0;
    float best_value = ag_data(t, values[0]);
    for (int i = 1; i < n; i++) {
        float value = ag_data(t, values[i]);
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return best;
}

float mt_mean(const float *values, int n) {
    if (!values || n <= 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        total += values[i];
    }
    return total / (float)n;
}

float mt_accuracy_binary(const float *pred, const float *target, int n, float threshold) {
    if (!pred || !target || n <= 0) {
        return 0.0f;
    }

    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred_class = pred[i] >= threshold ? 1 : 0;
        int target_class = target[i] >= 0.5f ? 1 : 0;
        if (pred_class == target_class) {
            correct++;
        }
    }
    return (float)correct / (float)n;
}

float mt_accuracy_multiclass(const int *pred, const int *target, int n) {
    if (!pred || !target || n <= 0) {
        return 0.0f;
    }

    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (pred[i] == target[i]) {
            correct++;
        }
    }
    return (float)correct / (float)n;
}

void mt_confusion_matrix(const int *pred, const int *target, int n, int n_classes, int *matrix) {
    if (!matrix || n_classes <= 0) {
        return;
    }

    for (int i = 0; i < n_classes * n_classes; i++) {
        matrix[i] = 0;
    }

    if (!pred || !target || n <= 0) {
        return;
    }

    for (int i = 0; i < n; i++) {
        int truth = target[i];
        int guess = pred[i];
        if (truth >= 0 && truth < n_classes && guess >= 0 && guess < n_classes) {
            matrix[truth * n_classes + guess]++;
        }
    }
}

int mt_eval_binary_linear(AgTape *t,
                          const MtLinear *model,
                          int graph_checkpoint,
                          const MtDataset *dataset,
                          float threshold,
                          MtEvalResult *result) {
    if (!t || !model || !dataset || !result || model->out_features != 1) {
        return 0;
    }
    if (dataset->n_samples <= 0 || dataset->n_features != model->in_features) {
        return 0;
    }

    float total_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < dataset->n_samples; i++) {
        ag_rewind(t, graph_checkpoint);

        AgVal input[model->in_features];
        for (int f = 0; f < model->in_features; f++) {
            input[f] = ag_leaf(t, mt_dataset_feature(dataset, i, f));
        }

        AgVal logits[1];
        AgVal pred[1];
        AgVal target[1];
        mt_linear_forward(t, model, input, logits);
        mt_sigmoid(t, logits, 1, pred);
        target[0] = ag_leaf(t, mt_dataset_label(dataset, i));

        AgVal loss = mt_bce_loss(t, pred, target, 1);
        total_loss += ag_data(t, loss);

        int pred_class = ag_data(t, pred[0]) >= threshold ? 1 : 0;
        int target_class = mt_dataset_label(dataset, i) >= 0.5f ? 1 : 0;
        if (pred_class == target_class) {
            correct++;
        }
    }

    result->loss_mean = total_loss / (float)dataset->n_samples;
    result->accuracy = (float)correct / (float)dataset->n_samples;
    result->correct = correct;
    result->total = dataset->n_samples;
    ag_rewind(t, graph_checkpoint);
    return 1;
}

int mt_eval_multiclass_linear(AgTape *t,
                              const MtLinear *model,
                              int graph_checkpoint,
                              const MtDataset *dataset,
                              int n_classes,
                              MtEvalResult *result,
                              int *confusion_matrix) {
    if (!t || !model || !dataset || !result || n_classes <= 0) {
        return 0;
    }
    if (dataset->n_samples <= 0 || dataset->n_features != model->in_features || model->out_features != n_classes) {
        return 0;
    }

    if (confusion_matrix) {
        for (int i = 0; i < n_classes * n_classes; i++) {
            confusion_matrix[i] = 0;
        }
    }

    float total_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < dataset->n_samples; i++) {
        ag_rewind(t, graph_checkpoint);

        AgVal input[model->in_features];
        for (int f = 0; f < model->in_features; f++) {
            input[f] = ag_leaf(t, mt_dataset_feature(dataset, i, f));
        }

        AgVal logits[n_classes];
        AgVal target[n_classes];
        int label = (int)mt_dataset_label(dataset, i);

        mt_linear_forward(t, model, input, logits);
        for (int c = 0; c < n_classes; c++) {
            target[c] = ag_leaf(t, c == label ? 1.0f : 0.0f);
        }

        AgVal loss = mt_cross_entropy_from_logits(t, logits, target, n_classes);
        int pred = mt_argmax(t, logits, n_classes);
        total_loss += ag_data(t, loss);

        if (pred == label) {
            correct++;
        }
        if (confusion_matrix && label >= 0 && label < n_classes && pred >= 0 && pred < n_classes) {
            confusion_matrix[label * n_classes + pred]++;
        }
    }

    result->loss_mean = total_loss / (float)dataset->n_samples;
    result->accuracy = (float)correct / (float)dataset->n_samples;
    result->correct = correct;
    result->total = dataset->n_samples;
    ag_rewind(t, graph_checkpoint);
    return 1;
}

static int mt_model_max_int(int a, int b) {
    return a > b ? a : b;
}

MtModel *mt_model_create_linear_binary(AgTape *t, int input_size) {
    if (!t || input_size <= 0) {
        return NULL;
    }

    MtModel *model = (MtModel *)calloc(1, sizeof(MtModel));
    if (!model) {
        return NULL;
    }

    model->kind = MT_MODEL_LINEAR_BINARY;
    model->input_size = input_size;
    model->hidden_size = 0;
    model->output_size = 1;
    model->linear = mt_linear_create(t, input_size, 1, 1);
    if (!model->linear) {
        mt_model_free(model);
        return NULL;
    }

    mt_linear_init_xavier_uniform(t, model->linear);
    return model;
}

MtModel *mt_model_create_mlp_binary(AgTape *t, int input_size, int hidden_size) {
    if (!t || input_size <= 0 || hidden_size <= 0) {
        return NULL;
    }

    MtModel *model = (MtModel *)calloc(1, sizeof(MtModel));
    if (!model) {
        return NULL;
    }

    model->kind = MT_MODEL_MLP_BINARY;
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->output_size = 1;
    model->hidden = mt_linear_create(t, input_size, hidden_size, 1);
    model->output = mt_linear_create(t, hidden_size, 1, 1);
    model->seq = mt_sequential_create(mt_model_max_int(input_size, hidden_size));

    if (!model->hidden || !model->output || !model->seq) {
        mt_model_free(model);
        return NULL;
    }

    mt_linear_init_xavier_uniform(t, model->hidden);
    mt_linear_init_xavier_uniform(t, model->output);
    if (!mt_sequential_add_linear(model->seq, model->hidden) ||
        !mt_sequential_add_activation(model->seq, MT_ACT_TANH) ||
        !mt_sequential_add_linear(model->seq, model->output) ||
        !mt_sequential_add_activation(model->seq, MT_ACT_SIGMOID)) {
        mt_model_free(model);
        return NULL;
    }

    return model;
}

void mt_model_free(MtModel *model) {
    if (!model) {
        return;
    }

    mt_sequential_free(model->seq);
    mt_linear_free(model->linear);
    mt_linear_free(model->hidden);
    mt_linear_free(model->output);
    free(model);
}

int mt_model_forward(AgTape *t,
                     const MtModel *model,
                     const AgVal *input,
                     int input_size,
                     AgVal *output,
                     int output_size) {
    if (!t || !model || !input || !output || input_size != model->input_size || output_size != model->output_size) {
        return 0;
    }

    if (model->kind == MT_MODEL_LINEAR_BINARY) {
        if (!model->linear || output_size != 1) {
            return 0;
        }
        AgVal logits[1];
        mt_linear_forward(t, model->linear, input, logits);
        mt_sigmoid(t, logits, 1, output);
        return 1;
    }

    if (model->kind == MT_MODEL_MLP_BINARY) {
        return mt_sequential_forward(t, model->seq, input, input_size, output, output_size);
    }

    return 0;
}

int mt_model_eval_binary(AgTape *t,
                         const MtModel *model,
                         int graph_checkpoint,
                         const MtDataset *dataset,
                         float threshold,
                         MtEvalResult *result) {
    if (!t || !model || !dataset || !result || model->output_size != 1) {
        return 0;
    }
    if (dataset->n_samples <= 0 || dataset->n_features != model->input_size) {
        return 0;
    }

    AgVal *input = (AgVal *)malloc(sizeof(AgVal) * (size_t)model->input_size);
    if (!input) {
        return 0;
    }

    float total_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < dataset->n_samples; i++) {
        ag_rewind(t, graph_checkpoint);

        for (int f = 0; f < model->input_size; f++) {
            input[f] = ag_leaf(t, mt_dataset_feature(dataset, i, f));
        }

        AgVal pred[1];
        AgVal target[1] = {
            ag_leaf(t, mt_dataset_label(dataset, i))
        };

        if (!mt_model_forward(t, model, input, model->input_size, pred, 1)) {
            free(input);
            ag_rewind(t, graph_checkpoint);
            return 0;
        }

        AgVal loss = mt_bce_loss(t, pred, target, 1);
        float prob = ag_data(t, pred[0]);
        int pred_class = prob >= threshold ? 1 : 0;
        int truth = mt_dataset_label(dataset, i) >= 0.5f ? 1 : 0;

        total_loss += ag_data(t, loss);
        if (pred_class == truth) {
            correct++;
        }
    }

    free(input);
    result->loss_mean = total_loss / (float)dataset->n_samples;
    result->accuracy = (float)correct / (float)dataset->n_samples;
    result->correct = correct;
    result->total = dataset->n_samples;
    ag_rewind(t, graph_checkpoint);
    return 1;
}

int mt_model_predict(AgTape *t,
                     const MtModel *model,
                     int graph_checkpoint,
                     const float *features,
                     float *output,
                     int output_size) {
    if (!t || !model || !features || !output || output_size != model->output_size) {
        return 0;
    }

    AgVal *input = (AgVal *)malloc(sizeof(AgVal) * (size_t)model->input_size);
    AgVal *pred = (AgVal *)malloc(sizeof(AgVal) * (size_t)output_size);
    if (!input || !pred) {
        free(input);
        free(pred);
        return 0;
    }

    ag_rewind(t, graph_checkpoint);
    for (int i = 0; i < model->input_size; i++) {
        input[i] = ag_leaf(t, features[i]);
    }

    int ok = mt_model_forward(t, model, input, model->input_size, pred, output_size);
    if (ok) {
        for (int i = 0; i < output_size; i++) {
            output[i] = ag_data(t, pred[i]);
        }
    }

    free(input);
    free(pred);
    ag_rewind(t, graph_checkpoint);
    return ok;
}

int mt_model_predict_dataset(AgTape *t,
                             const MtModel *model,
                             int graph_checkpoint,
                             const MtDataset *dataset,
                             float *outputs,
                             int output_size) {
    if (!t || !model || !dataset || !outputs || output_size != model->output_size) {
        return 0;
    }
    if (dataset->n_samples <= 0 || dataset->n_features != model->input_size) {
        return 0;
    }

    float *features = (float *)malloc(sizeof(float) * (size_t)dataset->n_features);
    if (!features) {
        return 0;
    }

    for (int sample = 0; sample < dataset->n_samples; sample++) {
        for (int feature = 0; feature < dataset->n_features; feature++) {
            features[feature] = mt_dataset_feature(dataset, sample, feature);
        }
        if (!mt_model_predict(t, model, graph_checkpoint, features, &outputs[sample * output_size], output_size)) {
            free(features);
            return 0;
        }
    }

    free(features);
    return 1;
}

static int mt_model_save_linear_block(FILE *file, const AgTape *t, const MtLinear *layer) {
    if (!file || !t || !layer) {
        return 0;
    }

    fprintf(file, "LINEAR %d %d %d\n", layer->in_features, layer->out_features, layer->use_bias);
    fprintf(file, "WEIGHTS %d\n", layer->in_features * layer->out_features);
    for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
        for (int in_idx = 0; in_idx < layer->in_features; in_idx++) {
            fprintf(file, "%.9g\n", ag_data(t, mt_linear_weight(layer, out_idx, in_idx)));
        }
    }

    fprintf(file, "BIAS %d\n", layer->use_bias ? layer->out_features : 0);
    if (layer->use_bias) {
        for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
            fprintf(file, "%.9g\n", ag_data(t, mt_linear_bias(layer, out_idx)));
        }
    }
    fprintf(file, "END_LINEAR\n");
    return ferror(file) == 0;
}

static int mt_model_load_linear_block(FILE *file, AgTape *t, MtLinear *layer) {
    if (!file || !t || !layer) {
        return 0;
    }

    char token[32];
    int in_features = 0;
    int out_features = 0;
    int use_bias = 0;
    int count = 0;
    float value = 0.0f;

    if (fscanf(file, "%31s %d %d %d", token, &in_features, &out_features, &use_bias) != 4 ||
        strcmp(token, "LINEAR") != 0 ||
        in_features != layer->in_features ||
        out_features != layer->out_features ||
        use_bias != layer->use_bias) {
        return 0;
    }

    if (fscanf(file, "%31s %d", token, &count) != 2 ||
        strcmp(token, "WEIGHTS") != 0 ||
        count != layer->in_features * layer->out_features) {
        return 0;
    }

    for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
        for (int in_idx = 0; in_idx < layer->in_features; in_idx++) {
            if (fscanf(file, "%f", &value) != 1) {
                return 0;
            }
            mt_linear_set_weight(t, layer, out_idx, in_idx, value);
        }
    }

    if (fscanf(file, "%31s %d", token, &count) != 2 ||
        strcmp(token, "BIAS") != 0 ||
        count != (layer->use_bias ? layer->out_features : 0)) {
        return 0;
    }

    if (layer->use_bias) {
        for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
            if (fscanf(file, "%f", &value) != 1) {
                return 0;
            }
            mt_linear_set_bias(t, layer, out_idx, value);
        }
    }

    return fscanf(file, "%31s", token) == 1 && strcmp(token, "END_LINEAR") == 0;
}

int mt_model_save(const AgTape *t, const MtModel *model, const char *path) {
    if (!t || !model || !path) {
        return 0;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return 0;
    }

    fprintf(file, "MINITORCH_MODEL_V1 %d %d %d %d\n",
            (int)model->kind,
            model->input_size,
            model->hidden_size,
            model->output_size);

    int ok = 0;
    if (model->kind == MT_MODEL_LINEAR_BINARY) {
        ok = mt_model_save_linear_block(file, t, model->linear);
    } else if (model->kind == MT_MODEL_MLP_BINARY) {
        ok = mt_model_save_linear_block(file, t, model->hidden) &&
             mt_model_save_linear_block(file, t, model->output);
    }

    if (ok) {
        fprintf(file, "END_MODEL\n");
        ok = ferror(file) == 0;
    }

    fclose(file);
    return ok;
}

int mt_model_load(AgTape *t, MtModel *model, const char *path) {
    if (!t || !model || !path) {
        return 0;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        return 0;
    }

    char token[32];
    int kind = -1;
    int input_size = 0;
    int hidden_size = 0;
    int output_size = 0;

    int ok = fscanf(file, "%31s %d %d %d %d", token, &kind, &input_size, &hidden_size, &output_size) == 5 &&
             strcmp(token, "MINITORCH_MODEL_V1") == 0 &&
             kind == (int)model->kind &&
             input_size == model->input_size &&
             hidden_size == model->hidden_size &&
             output_size == model->output_size;

    if (ok && model->kind == MT_MODEL_LINEAR_BINARY) {
        ok = mt_model_load_linear_block(file, t, model->linear);
    } else if (ok && model->kind == MT_MODEL_MLP_BINARY) {
        ok = mt_model_load_linear_block(file, t, model->hidden) &&
             mt_model_load_linear_block(file, t, model->output);
    } else {
        ok = 0;
    }

    if (ok) {
        ok = fscanf(file, "%31s", token) == 1 && strcmp(token, "END_MODEL") == 0;
    }

    fclose(file);
    return ok;
}
