#ifndef MINITORCH_SERIALIZATION_H
#define MINITORCH_SERIALIZATION_H

#include "minitorch/core/autograd.h"
#include "minitorch/nn/nn.h"

int mt_save_linear(const AgTape *t, const MtLinear *layer, const char *path);
int mt_load_linear(AgTape *t, MtLinear *layer, const char *path);

int mt_save_sequential(const AgTape *t, const MtSequential *seq, const char *path);
int mt_load_sequential(AgTape *t, MtSequential *seq, const char *path);

#endif /* MINITORCH_SERIALIZATION_H */
