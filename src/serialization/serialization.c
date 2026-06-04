#include "minitorch/serialization/serialization.h"

#include <stdio.h>
#include <string.h>

static int count_linear_layers(const MtSequential *seq) {
    if (!seq) {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < seq->n_layers; i++) {
        if (seq->layers[i].kind == MT_LAYER_LINEAR) {
            count++;
        }
    }
    return count;
}

static int save_linear_block(FILE *file, const AgTape *t, const MtLinear *layer) {
    if (!file || !t || !layer || !layer->weight) {
        return 0;
    }

    fprintf(file, "LINEAR %d %d %d\n", layer->in_features, layer->out_features, layer->use_bias);
    for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
        for (int in_idx = 0; in_idx < layer->in_features; in_idx++) {
            AgVal param = mt_linear_weight(layer, out_idx, in_idx);
            fprintf(file, "W %d %d %.9g\n", out_idx, in_idx, ag_data(t, param));
        }
    }

    if (layer->use_bias) {
        for (int out_idx = 0; out_idx < layer->out_features; out_idx++) {
            AgVal param = mt_linear_bias(layer, out_idx);
            fprintf(file, "B %d %.9g\n", out_idx, ag_data(t, param));
        }
    }

    fprintf(file, "END_LINEAR\n");
    return ferror(file) ? 0 : 1;
}

static int load_linear_block(FILE *file, AgTape *t, MtLinear *layer) {
    if (!file || !t || !layer || !layer->weight) {
        return 0;
    }

    char tag[32];
    int in_features = 0;
    int out_features = 0;
    int use_bias = 0;

    if (fscanf(file, "%31s %d %d %d", tag, &in_features, &out_features, &use_bias) != 4) {
        return 0;
    }
    if (strcmp(tag, "LINEAR") != 0 || in_features != layer->in_features ||
        out_features != layer->out_features || use_bias != layer->use_bias) {
        return 0;
    }

    int n_weights = layer->in_features * layer->out_features;
    for (int i = 0; i < n_weights; i++) {
        int out_idx = 0;
        int in_idx = 0;
        float value = 0.0f;
        if (fscanf(file, "%31s %d %d %f", tag, &out_idx, &in_idx, &value) != 4) {
            return 0;
        }
        if (strcmp(tag, "W") != 0) {
            return 0;
        }
        if (out_idx < 0 || out_idx >= layer->out_features ||
            in_idx < 0 || in_idx >= layer->in_features) {
            return 0;
        }
        mt_linear_set_weight(t, layer, out_idx, in_idx, value);
    }

    if (layer->use_bias) {
        for (int i = 0; i < layer->out_features; i++) {
            int out_idx = 0;
            float value = 0.0f;
            if (fscanf(file, "%31s %d %f", tag, &out_idx, &value) != 3) {
                return 0;
            }
            if (strcmp(tag, "B") != 0) {
                return 0;
            }
            if (out_idx < 0 || out_idx >= layer->out_features) {
                return 0;
            }
            mt_linear_set_bias(t, layer, out_idx, value);
        }
    }

    if (fscanf(file, "%31s", tag) != 1) {
        return 0;
    }
    return strcmp(tag, "END_LINEAR") == 0;
}

int mt_save_linear(const AgTape *t, const MtLinear *layer, const char *path) {
    if (!t || !layer || !path) {
        return 0;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return 0;
    }

    fprintf(file, "MINITORCH_LINEAR_V1\n");
    int ok = save_linear_block(file, t, layer);
    fclose(file);
    return ok;
}

int mt_load_linear(AgTape *t, MtLinear *layer, const char *path) {
    if (!t || !layer || !path) {
        return 0;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        return 0;
    }

    char header[64];
    int ok = fscanf(file, "%63s", header) == 1 && strcmp(header, "MINITORCH_LINEAR_V1") == 0;
    if (ok) {
        ok = load_linear_block(file, t, layer);
    }

    fclose(file);
    return ok;
}

int mt_save_sequential(const AgTape *t, const MtSequential *seq, const char *path) {
    if (!t || !seq || !path) {
        return 0;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return 0;
    }

    int n_linear = count_linear_layers(seq);

    fprintf(file, "MINITORCH_SEQUENTIAL_V1 %d\n", n_linear);
    int ok = 1;
    for (int i = 0; i < seq->n_layers; i++) {
        if (seq->layers[i].kind == MT_LAYER_LINEAR) {
            ok = save_linear_block(file, t, seq->layers[i].linear);
            if (!ok) {
                break;
            }
        }
    }

    fclose(file);
    return ok;
}

int mt_load_sequential(AgTape *t, MtSequential *seq, const char *path) {
    if (!t || !seq || !path) {
        return 0;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        return 0;
    }

    char header[64];
    int n_linear = 0;
    int ok = fscanf(file, "%63s %d", header, &n_linear) == 2 &&
             strcmp(header, "MINITORCH_SEQUENTIAL_V1") == 0;
    if (ok && n_linear != count_linear_layers(seq)) {
        ok = 0;
    }

    int layer_idx = 0;
    for (int loaded = 0; ok && loaded < n_linear; loaded++) {
        while (layer_idx < seq->n_layers && seq->layers[layer_idx].kind != MT_LAYER_LINEAR) {
            layer_idx++;
        }
        if (layer_idx >= seq->n_layers) {
            ok = 0;
            break;
        }
        ok = load_linear_block(file, t, seq->layers[layer_idx].linear);
        layer_idx++;
    }

    fclose(file);
    return ok;
}
