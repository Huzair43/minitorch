#ifndef MINITORCH_NN_H
#define MINITORCH_NN_H

#include "minitorch/core/autograd.h"

#ifndef MT_MAX_SEQ_LAYERS
#  define MT_MAX_SEQ_LAYERS 16
#endif

typedef enum {
    MT_ACT_RELU = 0,
    MT_ACT_SIGMOID,
    MT_ACT_TANH
} MtActivation;

typedef struct {
    int in_features;
    int out_features;
    int use_bias;
    AgVal *weight;
    AgVal *bias;
} MtLinear;

typedef enum {
    MT_LAYER_LINEAR = 0,
    MT_LAYER_ACTIVATION
} MtLayerKind;

typedef struct {
    MtLayerKind kind;
    MtLinear *linear;
    MtActivation activation;
} MtLayer;

typedef struct {
    MtLayer layers[MT_MAX_SEQ_LAYERS];
    int n_layers;
    int max_width;
    AgVal *buffer_a;
    AgVal *buffer_b;
} MtSequential;

MtLinear *mt_linear_create(AgTape *t, int in_features, int out_features, int use_bias);
void      mt_linear_free(MtLinear *layer);
void      mt_linear_init_uniform(AgTape *t, MtLinear *layer, float lo, float hi);
void      mt_linear_set_weight(AgTape *t, MtLinear *layer, int out_idx, int in_idx, float value);
void      mt_linear_set_bias(AgTape *t, MtLinear *layer, int out_idx, float value);
AgVal     mt_linear_weight(const MtLinear *layer, int out_idx, int in_idx);
AgVal     mt_linear_bias(const MtLinear *layer, int out_idx);
void      mt_linear_forward(AgTape *t, const MtLinear *layer, const AgVal *input, AgVal *output);
void      mt_linear_step(AgTape *t, MtLinear *layer, float lr);

void      mt_activation_forward(AgTape *t, MtActivation activation, const AgVal *input, int n, AgVal *output);
void      mt_relu(AgTape *t, const AgVal *input, int n, AgVal *output);
void      mt_sigmoid(AgTape *t, const AgVal *input, int n, AgVal *output);
void      mt_tanh(AgTape *t, const AgVal *input, int n, AgVal *output);
void      mt_softmax(AgTape *t, const AgVal *logits, int n, AgVal *probs);

MtSequential *mt_sequential_create(int max_width);
void          mt_sequential_free(MtSequential *seq);
int           mt_sequential_add_linear(MtSequential *seq, MtLinear *layer);
int           mt_sequential_add_activation(MtSequential *seq, MtActivation activation);
int           mt_sequential_forward(AgTape *t,
                                    const MtSequential *seq,
                                    const AgVal *input,
                                    int input_size,
                                    AgVal *output,
                                    int output_size);
void          mt_sequential_step(AgTape *t, MtSequential *seq, float lr);

AgVal     mt_mse_loss(AgTape *t, const AgVal *pred, const AgVal *target, int n);
AgVal     mt_bce_loss(AgTape *t, const AgVal *pred, const AgVal *target, int n);
AgVal     mt_cross_entropy_loss(AgTape *t, const AgVal *probs, const AgVal *target, int n);

#endif /* MINITORCH_NN_H */
