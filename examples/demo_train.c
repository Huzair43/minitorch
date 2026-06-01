#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minitorch/core/autograd.h"

#define MAX_SAMPLES 64
#define MAX_FEATURES 8

/* ── Dataset saisi par l'utilisateur ──────────────────── */
static float X[MAX_SAMPLES][MAX_FEATURES];
static float Y[MAX_SAMPLES];
static int   n_samples  = 0;
static int   n_features = 0;

/* ── Poids ─────────────────────────────────────────────── */
/* couche cachee : n_features -> 4 neurones                 */
/* sortie        : 4          -> 1                          */
#define H 4
static float wh[MAX_FEATURES][H], bh[H];
static float wo[H], bo;

static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void init_weights(void) {
    float scale = sqrtf(2.0f / (float)n_features);
    for (int i = 0; i < n_features; i++)
        for (int j = 0; j < H; j++)
            wh[i][j] = randf(-scale, scale);
    for (int j = 0; j < H; j++) { bh[j] = 0.0f; wo[j] = randf(-1.0f, 1.0f); }
    bo = 0.0f;
}

/* ── Saisie du dataset ─────────────────────────────────── */
static void input_dataset(void) {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║            Saisie du dataset                ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    do {
        printf("Nombre de features (1 a %d) : ", MAX_FEATURES);
        scanf("%d", &n_features);
    } while (n_features < 1 || n_features > MAX_FEATURES);

    do {
        printf("Nombre de samples  (2 a %d) : ", MAX_SAMPLES);
        scanf("%d", &n_samples);
    } while (n_samples < 2 || n_samples > MAX_SAMPLES);

    printf("\nLabel y : valeur reelle entre 0 et 1 (classification binaire).\n\n");

    for (int i = 0; i < n_samples; i++) {
        printf("Sample %d :\n", i + 1);
        for (int f = 0; f < n_features; f++) {
            printf("  x%d = ", f + 1);
            scanf("%f", &X[i][f]);
        }
        float y;
        do {
            printf("  y  = ");
            scanf("%f", &y);
            if (y < 0.0f || y > 1.0f)
                printf("  [erreur] y doit etre dans [0, 1]\n");
        } while (y < 0.0f || y > 1.0f);
        Y[i] = y;
        printf("\n");
    }
}

/* ── Affiche le recap du dataset ───────────────────────── */
static void print_dataset(void) {
    printf("Dataset (%d samples, %d features) :\n\n", n_samples, n_features);
    printf("  ");
    for (int f = 0; f < n_features; f++) printf("  x%-4d", f + 1);
    printf("  y\n  ");
    for (int f = 0; f < n_features + 1; f++) printf("───────");
    printf("\n");
    for (int i = 0; i < n_samples; i++) {
        printf("  ");
        for (int f = 0; f < n_features; f++) printf("  %-5.2f", X[i][f]);
        printf("  %.1f\n", Y[i]);
    }
    printf("\n");
}

/* ── Forward + backward sur un sample ─────────────────── */
static float forward_backward(int idx,
    float dwh[MAX_FEATURES][H], float dbh[H],
    float dwo[H], float *dbo)
{
    AgTape *t = ag_tape_create();

    /* inputs */
    AgVal vx[MAX_FEATURES];
    for (int f = 0; f < n_features; f++)
        vx[f] = ag_leaf(t, X[idx][f]);
    AgVal vy = ag_leaf(t, Y[idx]);

    /* poids couche cachee */
    AgVal vwh[MAX_FEATURES][H], vbh[H];
    for (int f = 0; f < n_features; f++)
        for (int j = 0; j < H; j++)
            vwh[f][j] = ag_leaf(t, wh[f][j]);
    for (int j = 0; j < H; j++)
        vbh[j] = ag_leaf(t, bh[j]);

    /* poids sortie */
    AgVal vwo[H];
    for (int j = 0; j < H; j++)
        vwo[j] = ag_leaf(t, wo[j]);
    AgVal vbo = ag_leaf(t, bo);

    /* ── Forward couche cachee ── */
    AgVal h[H];
    for (int j = 0; j < H; j++) {
        AgVal pre = vbh[j];
        for (int f = 0; f < n_features; f++)
            pre = ag_add(t, pre, ag_mul(t, vx[f], vwh[f][j]));
        h[j] = ag_tanh(t, pre);
    }

    /* ── Forward sortie ── */
    AgVal pre_out = vbo;
    for (int j = 0; j < H; j++)
        pre_out = ag_add(t, pre_out, ag_mul(t, h[j], vwo[j]));
    AgVal out = ag_sigmoid(t, pre_out);

    /* ── BCE loss ── */
    AgVal eps        = ag_leaf(t, 1e-7f);
    AgVal one        = ag_leaf(t, 1.0f);
    AgVal log_out    = ag_log(t, ag_add(t, out, eps));
    AgVal log_1_out  = ag_log(t, ag_add(t, ag_sub(t, one, out), eps));
    AgVal loss       = ag_neg(t,
                           ag_add(t,
                               ag_mul(t, vy, log_out),
                               ag_mul(t, ag_sub(t, one, vy), log_1_out)));

    float loss_val = ag_data(t, loss);

    /* ── Backward ── */
    ag_backward(t, loss);

    /* ── Lecture gradients ── */
    for (int f = 0; f < n_features; f++)
        for (int j = 0; j < H; j++)
            dwh[f][j] = ag_grad(t, vwh[f][j]);
    for (int j = 0; j < H; j++) {
        dbh[j] = ag_grad(t, vbh[j]);
        dwo[j] = ag_grad(t, vwo[j]);
    }
    *dbo = ag_grad(t, vbo);

    ag_tape_free(t);
    return loss_val;
}

/* ── Inference ─────────────────────────────────────────── */
static float predict(int idx) {
    float h[H];
    for (int j = 0; j < H; j++) {
        float pre = bh[j];
        for (int f = 0; f < n_features; f++)
            pre += X[idx][f] * wh[f][j];
        h[j] = tanhf(pre);
    }
    float pre_out = bo;
    for (int j = 0; j < H; j++)
        pre_out += h[j] * wo[j];
    return 1.0f / (1.0f + expf(-pre_out));
}

/* ── Boucle d'entrainement ─────────────────────────────── */
static void train(int epochs, float lr) {
    float dwh[MAX_FEATURES][H], dbh[H], dwo[H], dbo;

    printf("%-8s  %-12s\n", "Epoch", "Loss moy.");
    printf("────────  ────────────\n");

    int print_every = epochs / 10;
    if (print_every < 1) print_every = 1;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;

        for (int i = 0; i < n_samples; i++) {
            float loss = forward_backward(i, dwh, dbh, dwo, &dbo);
            total_loss += loss;

            /* SGD step */
            for (int f = 0; f < n_features; f++)
                for (int j = 0; j < H; j++)
                    wh[f][j] -= lr * dwh[f][j];
            for (int j = 0; j < H; j++) {
                bh[j] -= lr * dbh[j];
                wo[j] -= lr * dwo[j];
            }
            bo -= lr * dbo;
        }

        if (epoch % print_every == 0 || epoch == 1)
            printf("%-8d  %.6f\n", epoch, total_loss / n_samples);
    }
}

/* ── Resultats finaux ──────────────────────────────────── */
static void print_results(float threshold) {
    printf("\n────────────────────────────────────────────────────\n");
    printf("Predictions finales (seuil = %.2f) :\n\n", threshold);

    printf("  ");
    for (int f = 0; f < n_features; f++) printf("  x%-4d", f + 1);
    printf("  y_true  pred    classe\n  ");
    for (int f = 0; f < n_features + 3; f++) printf("───────");
    printf("\n");

    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        float p   = predict(i);
        int   cls = p >= threshold ? 1 : 0;
        int   ok  = cls == (int)roundf(Y[i]);
        correct  += ok;
        printf("  ");
        for (int f = 0; f < n_features; f++) printf("  %-5.2f", X[i][f]);
        printf("  %-6.1f  %-6.4f  %d  %s\n",
               Y[i], p, cls, ok ? "OK" : "FAUX");
    }
    printf("\nAccuracy : %d / %d  (%.0f%%)\n",
           correct, n_samples, 100.0f * correct / n_samples);
    printf("────────────────────────────────────────────────────\n");
}

/* ── Main ──────────────────────────────────────────────── */
int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   MiniTorch — Training demo  (MLP n-4-1)   ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    /* 1. Saisie du dataset */
    input_dataset();
    print_dataset();

    /* 2. Hyperparametres */
    int   epochs;
    float lr, threshold;

    printf("Hyperparametres :\n");
    printf("  Epochs    : "); scanf("%d",  &epochs);
    printf("  LR        : "); scanf("%f",  &lr);
    printf("  Seuil     : "); scanf("%f",  &threshold);
    printf("\n");

    /* 3. Init + train */
    srand(42);
    init_weights();
    train(epochs, lr);

    /* 4. Resultats */
    print_results(threshold);

    return 0;
}
