#ifndef MINITORCH_DATASET_H
#define MINITORCH_DATASET_H

typedef struct {
    int n_samples;
    int n_features;
    float *x;
    float *y;
    int *indices;
} MtDataset;

typedef struct {
    int batch_size;
    int n_features;
    float *x;
    float *y;
} MtBatch;

typedef struct {
    int n_rows;
    int n_columns;
    int n_features;
    int has_header;
    int n_classes;
    int is_classification;
    int is_binary;
    float label_min;
    float label_max;
} MtCsvInfo;

MtDataset *mt_dataset_create(const float *x, const float *y, int n_samples, int n_features);
int        mt_dataset_analyze_csv(const char *path, int has_header, MtCsvInfo *info);
MtDataset *mt_dataset_load_csv(const char *path, int n_features, int has_header);
MtDataset *mt_dataset_load_csv_auto(const char *path, int has_header, MtCsvInfo *info);
void       mt_dataset_free(MtDataset *dataset);
void       mt_dataset_shuffle(MtDataset *dataset);
int        mt_dataset_split(const MtDataset *dataset,
                            float train_ratio,
                            float val_ratio,
                            MtDataset **train,
                            MtDataset **val,
                            MtDataset **test);
int        mt_dataset_num_batches(const MtDataset *dataset, int batch_size);
int        mt_dataset_get_batch(const MtDataset *dataset, int batch_index, int batch_size, MtBatch *batch);
float      mt_dataset_feature(const MtDataset *dataset, int sample_idx, int feature_idx);
float      mt_dataset_label(const MtDataset *dataset, int sample_idx);

MtBatch   *mt_batch_create(int batch_size, int n_features);
void       mt_batch_free(MtBatch *batch);
float      mt_batch_feature(const MtBatch *batch, int sample_idx, int feature_idx);
float      mt_batch_label(const MtBatch *batch, int sample_idx);

#endif /* MINITORCH_DATASET_H */
