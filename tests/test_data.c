#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "minitorch/data/dataset.h"

static int _passed = 0;
static int _failed = 0;

static void _check(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff <= tol) {
        printf("  [PASS] %-40s got=%.6f\n", name, got);
        _passed++;
    } else {
        printf("  [FAIL] %-40s got=%.6f  expected=%.6f  diff=%.2e\n",
               name, got, expected, diff);
        _failed++;
    }
}

#define CHECK(name, got, expected) _check(name, got, expected, 1e-4f)

static void section(const char *title) {
    printf("\n-- %s\n", title);
}

static void summary(void) {
    printf("\n==============================================\n");
    printf("  %d reussis  |  %d echoues  |  %d total\n",
           _passed, _failed, _passed + _failed);
    printf("==============================================\n");
}

static void test_dataset_access(void) {
    section("Dataset : acces aux donnees");

    float x[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    float y[] = {0.0f, 1.0f, 1.0f};
    MtDataset *dataset = mt_dataset_create(x, y, 3, 2);

    CHECK("n_samples", (float)dataset->n_samples, 3.0f);
    CHECK("n_features", (float)dataset->n_features, 2.0f);
    CHECK("x[1,0]", mt_dataset_feature(dataset, 1, 0), 3.0f);
    CHECK("x[2,1]", mt_dataset_feature(dataset, 2, 1), 6.0f);
    CHECK("y[1]", mt_dataset_label(dataset, 1), 1.0f);

    mt_dataset_free(dataset);
}

static void test_batch_extraction(void) {
    section("Batch : extraction");

    float x[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f
    };
    float y[] = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    MtDataset *dataset = mt_dataset_create(x, y, 5, 2);
    MtBatch *batch = mt_batch_create(2, 2);

    int n_batches = mt_dataset_num_batches(dataset, 2);
    int count0 = mt_dataset_get_batch(dataset, 0, 2, batch);
    CHECK("nombre de batches", (float)n_batches, 3.0f);
    CHECK("taille batch 0", (float)count0, 2.0f);
    CHECK("batch0 x[1,1]", mt_batch_feature(batch, 1, 1), 4.0f);
    CHECK("batch0 y[1]", mt_batch_label(batch, 1), 1.0f);

    int count2 = mt_dataset_get_batch(dataset, 2, 2, batch);
    CHECK("taille dernier batch", (float)count2, 1.0f);
    CHECK("dernier batch x[0,0]", mt_batch_feature(batch, 0, 0), 9.0f);
    CHECK("dernier batch y[0]", mt_batch_label(batch, 0), 1.0f);

    mt_batch_free(batch);
    mt_dataset_free(dataset);
}

static void test_invalid_inputs(void) {
    section("Dataset : entrees invalides");

    float x[] = {1.0f};
    float y[] = {0.0f};
    MtDataset *dataset = mt_dataset_create(x, y, 1, 1);
    MtBatch *wrong_batch = mt_batch_create(1, 2);

    CHECK("create invalide", mt_dataset_create(NULL, y, 1, 1) == NULL ? 1.0f : 0.0f, 1.0f);
    CHECK("batch dimensions incompatibles", (float)mt_dataset_get_batch(dataset, 0, 1, wrong_batch), 0.0f);

    mt_batch_free(wrong_batch);
    mt_dataset_free(dataset);
}

static void test_dataset_split(void) {
    section("Dataset : split train val test");

    float x[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
        5.0f, 50.0f
    };
    float y[] = {0.0f, 1.0f, 0.0f, 1.0f, 1.0f};

    MtDataset *dataset = mt_dataset_create(x, y, 5, 2);
    MtDataset *train = NULL;
    MtDataset *val = NULL;
    MtDataset *test = NULL;

    int ok = mt_dataset_split(dataset, 0.6f, 0.2f, &train, &val, &test);

    CHECK("split accepte", (float)ok, 1.0f);
    if (ok) {
        CHECK("train taille", (float)train->n_samples, 3.0f);
        CHECK("val taille", (float)val->n_samples, 1.0f);
        CHECK("test taille", (float)test->n_samples, 1.0f);
        CHECK("train x[2,0]", mt_dataset_feature(train, 2, 0), 3.0f);
        CHECK("val x[0,0]", mt_dataset_feature(val, 0, 0), 4.0f);
        CHECK("test y[0]", mt_dataset_label(test, 0), 1.0f);
    }

    mt_dataset_free(train);
    mt_dataset_free(val);
    mt_dataset_free(test);
    mt_dataset_free(dataset);
}

static void test_dataset_load_csv(void) {
    section("Dataset : chargement CSV");

    const char *path = "test_dataset_load.csv";
    FILE *file = fopen(path, "w");
    if (!file) {
        CHECK("creation fichier CSV", 0.0f, 1.0f);
        return;
    }

    fprintf(file, "x1,x2,y\n");
    fprintf(file, "1.0,2.0,0.0\n");
    fprintf(file, "3.5,4.5,1.0\n");
    fprintf(file, "5.0;6.0;1.0\n");
    fclose(file);

    MtDataset *dataset = mt_dataset_load_csv(path, 2, 1);

    CHECK("csv charge", dataset != NULL ? 1.0f : 0.0f, 1.0f);
    if (dataset) {
        CHECK("csv n_samples", (float)dataset->n_samples, 3.0f);
        CHECK("csv n_features", (float)dataset->n_features, 2.0f);
        CHECK("csv x[1,0]", mt_dataset_feature(dataset, 1, 0), 3.5f);
        CHECK("csv x[2,1]", mt_dataset_feature(dataset, 2, 1), 6.0f);
        CHECK("csv y[2]", mt_dataset_label(dataset, 2), 1.0f);
    }

    mt_dataset_free(dataset);
    remove(path);
}

int main(void) {
    printf("test_data : suite complete\n");

    test_dataset_access();
    test_batch_extraction();
    test_invalid_inputs();
    test_dataset_split();
    test_dataset_load_csv();

    summary();
    return _failed == 0 ? 0 : 1;
}
