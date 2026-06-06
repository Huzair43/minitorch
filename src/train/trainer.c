#include "minitorch/train/trainer.h"

#include <stdlib.h>

MtTrainConfig mt_train_config_default(void) {
    MtTrainConfig config;
    config.epochs = 100;
    config.batch_size = 4;
    config.shuffle = 1;
    config.print_every = 0;
    return config;
}

static int mt_trainer_normalize_config(const MtTrainConfig *config, MtTrainConfig *out) {
    if (!out) {
        return 0;
    }

    *out = config ? *config : mt_train_config_default();
    if (out->epochs <= 0) {
        out->epochs = 100;
    }
    if (out->batch_size <= 0) {
        out->batch_size = 4;
    }
    if (out->print_every < 0) {
        out->print_every = 0;
    }
    return 1;
}

static float mt_trainer_train_binary_batch(AgTape *t,
                                           MtModel *model,
                                           MtOptimizer *optim,
                                           const MtLoss *loss,
                                           int graph_checkpoint,
                                           const MtBatch *batch,
                                           int count) {
    if (!t || !model || !optim || !loss || !batch || count <= 0) {
        return 0.0f;
    }

    AgVal *input = (AgVal *)malloc(sizeof(AgVal) * (size_t)model->input_size);
    AgVal *pred = (AgVal *)malloc(sizeof(AgVal) * (size_t)count);
    AgVal *target = (AgVal *)malloc(sizeof(AgVal) * (size_t)count);
    if (!input || !pred || !target) {
        free(input);
        free(pred);
        free(target);
        return 0.0f;
    }

    ag_rewind(t, graph_checkpoint);
    for (int sample = 0; sample < count; sample++) {
        for (int feature = 0; feature < model->input_size; feature++) {
            input[feature] = ag_leaf(t, mt_batch_feature(batch, sample, feature));
        }
        if (!mt_model_forward(t, model, input, model->input_size, &pred[sample], 1)) {
            pred[sample] = ag_leaf(t, 0.5f);
        }
        target[sample] = ag_leaf(t, mt_batch_label(batch, sample));
    }

    AgVal batch_loss = mt_loss_forward(t, loss, pred, target, count);
    float loss_value = ag_data(t, batch_loss);

    mt_optimizer_zero_grad(t, optim);
    ag_backward(t, batch_loss);
    mt_optimizer_step(t, optim);

    free(input);
    free(pred);
    free(target);
    return loss_value;
}

int mt_trainer_train_binary(AgTape *t,
                            MtModel *model,
                            MtOptimizer *optim,
                            const MtLoss *loss,
                            int graph_checkpoint,
                            MtDataset *train,
                            const MtTrainConfig *config,
                            MtTrainHistory *history) {
    MtTrainConfig cfg;
    if (!t || !model || !optim || !loss || !train || train->n_samples <= 0) {
        return 0;
    }
    if (model->output_size != 1 || train->n_features != model->input_size) {
        return 0;
    }
    if (!mt_trainer_normalize_config(config, &cfg)) {
        return 0;
    }

    MtBatch *batch = mt_batch_create(cfg.batch_size, train->n_features);
    if (!batch) {
        return 0;
    }

    MtTrainHistory local_history;
    local_history.last_loss = 0.0f;
    local_history.epochs_ran = 0;
    local_history.batches_seen = 0;
    local_history.samples_seen = 0;

    for (int epoch = 1; epoch <= cfg.epochs; epoch++) {
        float total_loss = 0.0f;
        int seen = 0;
        int n_batches = mt_dataset_num_batches(train, cfg.batch_size);

        if (cfg.shuffle) {
            mt_dataset_shuffle(train);
        }

        for (int batch_index = 0; batch_index < n_batches; batch_index++) {
            int count = mt_dataset_get_batch(train, batch_index, cfg.batch_size, batch);
            if (count <= 0) {
                continue;
            }

            float batch_loss = mt_trainer_train_binary_batch(t, model, optim, loss, graph_checkpoint, batch, count);
            total_loss += batch_loss * (float)count;
            seen += count;
            local_history.batches_seen++;
            local_history.samples_seen += count;
        }

        if (seen > 0) {
            local_history.last_loss = total_loss / (float)seen;
        }
        local_history.epochs_ran = epoch;
    }

    if (history) {
        *history = local_history;
    }

    mt_batch_free(batch);
    ag_rewind(t, graph_checkpoint);
    return 1;
}
