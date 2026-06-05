#include "minitorch/data/dataset.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int mt_parse_csv_line(const char *line, int n_features, float *features, float *label) {
    if (!line || n_features <= 0 || !features || !label) {
        return 0;
    }

    const char *cursor = line;
    char *end = NULL;
    for (int i = 0; i < n_features; i++) {
        features[i] = strtof(cursor, &end);
        if (end == cursor) {
            return 0;
        }
        cursor = end;
        while (*cursor && isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor != ',' && *cursor != ';') {
            return 0;
        }
        cursor++;
    }

    *label = strtof(cursor, &end);
    return end != cursor;
}

static MtDataset *mt_dataset_slice(const MtDataset *dataset, int start, int count) {
    if (!dataset || start < 0 || count <= 0 || start + count > dataset->n_samples) {
        return NULL;
    }

    float *x = (float *)malloc(sizeof(float) * (size_t)(count * dataset->n_features));
    float *y = (float *)malloc(sizeof(float) * (size_t)count);
    if (!x || !y) {
        free(x);
        free(y);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        int src_idx = dataset->indices ? dataset->indices[start + i] : start + i;
        for (int f = 0; f < dataset->n_features; f++) {
            x[i * dataset->n_features + f] = mt_dataset_feature(dataset, src_idx, f);
        }
        y[i] = mt_dataset_label(dataset, src_idx);
    }

    MtDataset *slice = mt_dataset_create(x, y, count, dataset->n_features);
    free(x);
    free(y);
    return slice;
}

MtDataset *mt_dataset_create(const float *x, const float *y, int n_samples, int n_features) {
    if (!x || !y || n_samples <= 0 || n_features <= 0) {
        return NULL;
    }

    MtDataset *dataset = (MtDataset *)calloc(1, sizeof(MtDataset));
    if (!dataset) {
        return NULL;
    }

    dataset->n_samples = n_samples;
    dataset->n_features = n_features;
    dataset->x = (float *)malloc(sizeof(float) * (size_t)(n_samples * n_features));
    dataset->y = (float *)malloc(sizeof(float) * (size_t)n_samples);
    dataset->indices = (int *)malloc(sizeof(int) * (size_t)n_samples);
    if (!dataset->x || !dataset->y || !dataset->indices) {
        mt_dataset_free(dataset);
        return NULL;
    }

    memcpy(dataset->x, x, sizeof(float) * (size_t)(n_samples * n_features));
    memcpy(dataset->y, y, sizeof(float) * (size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        dataset->indices[i] = i;
    }

    return dataset;
}

MtDataset *mt_dataset_load_csv(const char *path, int n_features, int has_header) {
    if (!path || n_features <= 0) {
        return NULL;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        return NULL;
    }

    char line[1024];
    int count = 0;
    if (has_header) {
        fgets(line, sizeof(line), file);
    }
    while (fgets(line, sizeof(line), file)) {
        float *features = (float *)malloc(sizeof(float) * (size_t)n_features);
        float label = 0.0f;
        int ok = features && mt_parse_csv_line(line, n_features, features, &label);
        free(features);
        if (ok) {
            count++;
        }
    }

    if (count <= 0) {
        fclose(file);
        return NULL;
    }

    float *x = (float *)malloc(sizeof(float) * (size_t)(count * n_features));
    float *y = (float *)malloc(sizeof(float) * (size_t)count);
    if (!x || !y) {
        free(x);
        free(y);
        fclose(file);
        return NULL;
    }

    rewind(file);
    if (has_header) {
        fgets(line, sizeof(line), file);
    }

    int row = 0;
    while (row < count && fgets(line, sizeof(line), file)) {
        float *features = (float *)malloc(sizeof(float) * (size_t)n_features);
        float label = 0.0f;
        if (!features || !mt_parse_csv_line(line, n_features, features, &label)) {
            free(features);
            continue;
        }
        for (int f = 0; f < n_features; f++) {
            x[row * n_features + f] = features[f];
        }
        y[row] = label;
        free(features);
        row++;
    }

    fclose(file);
    MtDataset *dataset = mt_dataset_create(x, y, row, n_features);
    free(x);
    free(y);
    return dataset;
}

void mt_dataset_free(MtDataset *dataset) {
    if (!dataset) {
        return;
    }
    free(dataset->x);
    free(dataset->y);
    free(dataset->indices);
    free(dataset);
}

void mt_dataset_shuffle(MtDataset *dataset) {
    if (!dataset || !dataset->indices) {
        return;
    }

    for (int i = dataset->n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = dataset->indices[i];
        dataset->indices[i] = dataset->indices[j];
        dataset->indices[j] = tmp;
    }
}

int mt_dataset_split(const MtDataset *dataset,
                     float train_ratio,
                     float val_ratio,
                     MtDataset **train,
                     MtDataset **val,
                     MtDataset **test) {
    if (!train || !val || !test) {
        return 0;
    }
    *train = NULL;
    *val = NULL;
    *test = NULL;

    if (!dataset || dataset->n_samples < 3) {
        return 0;
    }
    if (train_ratio <= 0.0f || val_ratio < 0.0f || train_ratio + val_ratio >= 1.0f) {
        return 0;
    }

    int train_count = (int)((float)dataset->n_samples * train_ratio);
    int val_count = (int)((float)dataset->n_samples * val_ratio);
    int test_count = dataset->n_samples - train_count - val_count;

    if (train_count <= 0 || test_count <= 0) {
        return 0;
    }

    *train = mt_dataset_slice(dataset, 0, train_count);
    if (val_count > 0) {
        *val = mt_dataset_slice(dataset, train_count, val_count);
    }
    *test = mt_dataset_slice(dataset, train_count + val_count, test_count);

    if (!*train || (val_count > 0 && !*val) || !*test) {
        mt_dataset_free(*train);
        mt_dataset_free(*val);
        mt_dataset_free(*test);
        *train = NULL;
        *val = NULL;
        *test = NULL;
        return 0;
    }

    return 1;
}

int mt_dataset_num_batches(const MtDataset *dataset, int batch_size) {
    if (!dataset || batch_size <= 0) {
        return 0;
    }
    return (dataset->n_samples + batch_size - 1) / batch_size;
}

int mt_dataset_get_batch(const MtDataset *dataset, int batch_index, int batch_size, MtBatch *batch) {
    if (!dataset || !batch || !batch->x || !batch->y || batch_size <= 0 || batch_index < 0) {
        return 0;
    }
    if (batch->n_features != dataset->n_features) {
        return 0;
    }

    int start = batch_index * batch_size;
    if (start >= dataset->n_samples) {
        return 0;
    }

    int count = dataset->n_samples - start;
    if (count > batch_size) {
        count = batch_size;
    }
    if (count > batch->batch_size) {
        return 0;
    }

    for (int i = 0; i < count; i++) {
        int src_idx = dataset->indices ? dataset->indices[start + i] : start + i;
        for (int f = 0; f < dataset->n_features; f++) {
            batch->x[i * batch->n_features + f] = mt_dataset_feature(dataset, src_idx, f);
        }
        batch->y[i] = mt_dataset_label(dataset, src_idx);
    }

    return count;
}

float mt_dataset_feature(const MtDataset *dataset, int sample_idx, int feature_idx) {
    if (!dataset || !dataset->x) {
        return 0.0f;
    }
    if (sample_idx < 0 || sample_idx >= dataset->n_samples ||
        feature_idx < 0 || feature_idx >= dataset->n_features) {
        return 0.0f;
    }
    return dataset->x[sample_idx * dataset->n_features + feature_idx];
}

float mt_dataset_label(const MtDataset *dataset, int sample_idx) {
    if (!dataset || !dataset->y || sample_idx < 0 || sample_idx >= dataset->n_samples) {
        return 0.0f;
    }
    return dataset->y[sample_idx];
}

MtBatch *mt_batch_create(int batch_size, int n_features) {
    if (batch_size <= 0 || n_features <= 0) {
        return NULL;
    }

    MtBatch *batch = (MtBatch *)calloc(1, sizeof(MtBatch));
    if (!batch) {
        return NULL;
    }

    batch->batch_size = batch_size;
    batch->n_features = n_features;
    batch->x = (float *)malloc(sizeof(float) * (size_t)(batch_size * n_features));
    batch->y = (float *)malloc(sizeof(float) * (size_t)batch_size);
    if (!batch->x || !batch->y) {
        mt_batch_free(batch);
        return NULL;
    }

    return batch;
}

void mt_batch_free(MtBatch *batch) {
    if (!batch) {
        return;
    }
    free(batch->x);
    free(batch->y);
    free(batch);
}

float mt_batch_feature(const MtBatch *batch, int sample_idx, int feature_idx) {
    if (!batch || !batch->x) {
        return 0.0f;
    }
    if (sample_idx < 0 || sample_idx >= batch->batch_size ||
        feature_idx < 0 || feature_idx >= batch->n_features) {
        return 0.0f;
    }
    return batch->x[sample_idx * batch->n_features + feature_idx];
}

float mt_batch_label(const MtBatch *batch, int sample_idx) {
    if (!batch || !batch->y || sample_idx < 0 || sample_idx >= batch->batch_size) {
        return 0.0f;
    }
    return batch->y[sample_idx];
}
