#ifndef MINITORCH_TRAINER_H
#define MINITORCH_TRAINER_H

#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"

typedef struct {
    int epochs;
    int batch_size;
    int shuffle;
    int print_every;
} MtTrainConfig;

typedef struct {
    float last_loss;
    int epochs_ran;
    int batches_seen;
    int samples_seen;
} MtTrainHistory;

MtTrainConfig mt_train_config_default(void);

int mt_trainer_train_binary(AgTape *t,
                            MtModel *model,
                            MtOptimizer *optim,
                            const MtLoss *loss,
                            int graph_checkpoint,
                            MtDataset *train,
                            const MtTrainConfig *config,
                            MtTrainHistory *history);

int mt_trainer_train_multiclass(AgTape *t,
                                MtModel *model,
                                MtOptimizer *optim,
                                const MtLoss *loss,
                                int graph_checkpoint,
                                MtDataset *train,
                                const MtTrainConfig *config,
                                MtTrainHistory *history);

#endif /* MINITORCH_TRAINER_H */
