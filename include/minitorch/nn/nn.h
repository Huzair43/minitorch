#ifndef MINITORCH_NN_H
#define MINITORCH_NN_H

#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"

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

typedef struct {
    float loss_mean;
    float accuracy;
    int correct;
    int total;
} MtEvalResult;

typedef enum {
    MT_LOSS_MSE = 0,
    MT_LOSS_BCE,
    MT_LOSS_CROSS_ENTROPY,
    MT_LOSS_CROSS_ENTROPY_FROM_LOGITS
} MtLossKind;

typedef struct {
    MtLossKind kind;
} MtLoss;

typedef enum {
    MT_MODEL_LINEAR_BINARY = 0,
    MT_MODEL_MLP_BINARY,
    MT_MODEL_LINEAR_MULTICLASS,
    MT_MODEL_MLP_MULTICLASS
} MtModelKind;

typedef struct {
    MtModelKind kind;
    int input_size;
    int hidden_size;
    int output_size;
    MtLinear *linear;
    MtLinear *hidden;
    MtLinear *output;
    MtSequential *seq;
} MtModel;

MtLinear *mt_linear_create(AgTape *t, int in_features, int out_features, int use_bias);
void      mt_linear_free(MtLinear *layer);
void      mt_linear_init_uniform(AgTape *t, MtLinear *layer, float lo, float hi);
void      mt_linear_init_zeros(AgTape *t, MtLinear *layer);
void      mt_linear_init_xavier_uniform(AgTape *t, MtLinear *layer);
void      mt_linear_init_he_uniform(AgTape *t, MtLinear *layer);
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
AgVal     mt_cross_entropy_from_logits(AgTape *t, const AgVal *logits, const AgVal *target, int n);
MtLoss    mt_loss_create(MtLossKind kind);
AgVal     mt_loss_forward(AgTape *t, const MtLoss *loss, const AgVal *pred, const AgVal *target, int n);

int       mt_argmax_values(const float *values, int n);
int       mt_argmax(AgTape *t, const AgVal *values, int n);
float     mt_mean(const float *values, int n);
float     mt_accuracy_binary(const float *pred, const float *target, int n, float threshold);
float     mt_accuracy_multiclass(const int *pred, const int *target, int n);
void      mt_confusion_matrix(const int *pred, const int *target, int n, int n_classes, int *matrix);

int       mt_eval_binary_linear(AgTape *t,
                                const MtLinear *model,
                                int graph_checkpoint,
                                const MtDataset *dataset,
                                float threshold,
                                MtEvalResult *result);
int       mt_eval_multiclass_linear(AgTape *t,
                                    const MtLinear *model,
                                    int graph_checkpoint,
                                    const MtDataset *dataset,
                                    int n_classes,
                                    MtEvalResult *result,
                                    int *confusion_matrix);

MtModel  *mt_model_create_linear_binary(AgTape *t, int input_size);
MtModel  *mt_model_create_mlp_binary(AgTape *t, int input_size, int hidden_size);
MtModel  *mt_model_create_linear_multiclass(AgTape *t, int input_size, int n_classes);
MtModel  *mt_model_create_mlp_multiclass(AgTape *t, int input_size, int hidden_size, int n_classes);
void      mt_model_free(MtModel *model);
int       mt_model_forward(AgTape *t,
                           const MtModel *model,
                           const AgVal *input,
                           int input_size,
                           AgVal *output,
                           int output_size);
int       mt_model_eval_binary(AgTape *t,
                               const MtModel *model,
                               int graph_checkpoint,
                               const MtDataset *dataset,
                               float threshold,
                               MtEvalResult *result);
int       mt_model_eval_multiclass(AgTape *t,
                                   const MtModel *model,
                                   int graph_checkpoint,
                                   const MtDataset *dataset,
                                   MtEvalResult *result,
                                   int *confusion_matrix);
int       mt_model_predict(AgTape *t,
                           const MtModel *model,
                           int graph_checkpoint,
                           const float *features,
                           float *output,
                           int output_size);
int       mt_model_predict_dataset(AgTape *t,
                                   const MtModel *model,
                                   int graph_checkpoint,
                                   const MtDataset *dataset,
                                   float *outputs,
                                   int output_size);
int       mt_model_save(const AgTape *t, const MtModel *model, const char *path);
int       mt_model_load(AgTape *t, MtModel *model, const char *path);

#endif /* MINITORCH_NN_H */
