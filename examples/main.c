#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "minitorch/core/tensor.h"
#include "minitorch/core/tensor_ops.h"
#include "minitorch/core/tensor_linalg.h"
#include "minitorch/core/autograd.h"
#include "minitorch/data/dataset.h"
#include "minitorch/nn/nn.h"
#include "minitorch/optim/optim.h"
#include "minitorch/serialization/serialization.h"
#include "minitorch/train/trainer.h"

static void configure_console(void) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif
}

static void print_title(const char* title) {
    printf("\n=== %s ===\n", title);
}

static void print_tensor_2d(const Tensor* t) {
    if (!t || t->ndim != 2) {
        printf("<invalid tensor>\n");
        return;
    }

    for (int i = 0; i < t->shape[0]; i++) {
        printf("[");
        for (int j = 0; j < t->shape[1]; j++) {
            if (j > 0) printf(", ");
            printf("%7.3f", tensor_get(t, (int[]){i, j}));
        }
        printf(" ]\n");
    }
}

static void demo_tensor_basics(void) {
    print_title("Bases des tenseurs");

    int shape[2] = {2, 3};
    Tensor* a = tensor_create(shape, 2);
    Tensor* b = tensor_create(shape, 2);

    float a_values[] = {1, 2, 3, 4, 5, 6};
    float b_values[] = {10, 20, 30, 40, 50, 60};
    for (int i = 0; i < 6; i++) {
        a->data[i] = a_values[i];
        b->data[i] = b_values[i];
    }

    Tensor* sum = tensor_add(a, b);
    Tensor* prod = tensor_mul(a, b);

    printf("A:\n");
    print_tensor_2d(a);
    printf("B:\n");
    print_tensor_2d(b);
    printf("A + B:\n");
    print_tensor_2d(sum);
    printf("A * B:\n");
    print_tensor_2d(prod);

    tensor_free(a);
    tensor_free(b);
    tensor_free(sum);
    tensor_free(prod);
}

static void demo_broadcast_and_reduce(void) {
    print_title("Broadcast et réduction");

    int matrix_shape[2] = {2, 3};
    int bias_shape[1] = {3};
    Tensor* matrix = tensor_create(matrix_shape, 2);
    Tensor* bias = tensor_create(bias_shape, 1);

    float matrix_values[] = {1, 2, 3, 4, 5, 6};
    float bias_values[] = {10, 20, 30};
    for (int i = 0; i < 6; i++) matrix->data[i] = matrix_values[i];
    for (int i = 0; i < 3; i++) bias->data[i] = bias_values[i];

    Tensor* shifted = tensor_add(matrix, bias);
    Tensor* row_sum = tensor_sum(matrix, 1);
    Tensor* col_mean = tensor_mean(matrix, 0);
    Tensor* bias_broadcast = tensor_broadcast_to(bias, matrix_shape, 2);

    printf("Matrice :\n");
    print_tensor_2d(matrix);
    printf("Ligne de biais :\n");
    print_tensor_2d(bias_broadcast);
    printf("Matrice + biais :\n");
    print_tensor_2d(shifted);
    printf("Sommes par ligne (axe=1) :\n");
    tensor_print(row_sum);
    printf("Moyennes par colonne (axe=0) :\n");
    tensor_print(col_mean);

    tensor_free(matrix);
    tensor_free(bias);
    tensor_free(shifted);
    tensor_free(row_sum);
    tensor_free(col_mean);
    tensor_free(bias_broadcast);
}

static void demo_linalg(void) {
    print_title("Algèbre linéaire");

    int a_shape[2] = {2, 2};
    int b_shape[2] = {2, 2};
    Tensor* a = tensor_create(a_shape, 2);
    Tensor* b = tensor_create(b_shape, 2);

    float a_values[] = {1, 2, 3, 4};
    float b_values[] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) {
        a->data[i] = a_values[i];
        b->data[i] = b_values[i];
    }

    Tensor* c = tensor_matmul(a, b);
    Tensor* at = tensor_transpose(a, 0, 1);
    float det = tensor_det(a);

    printf("A:\n");
    print_tensor_2d(a);
    printf("B:\n");
    print_tensor_2d(b);
    printf("A @ B:\n");
    print_tensor_2d(c);
    printf("transposée(A) :\n");
    print_tensor_2d(at);
    printf("det(A) = %.3f\n", det);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(at);
}

static void demo_scalar_autograd(void) {
    print_title("Chaîne autograd scalaire");

    AgTape* tape = ag_tape_create();
    AgVal x = ag_leaf(tape, 2.0f);
    AgVal w = ag_leaf(tape, 3.0f);
    AgVal b = ag_leaf(tape, 1.0f);
    AgVal y = ag_leaf(tape, 5.0f);

    AgVal y_hat = ag_add(tape, ag_mul(tape, x, w), b);
    AgVal diff = ag_sub(tape, y_hat, y);
    AgVal loss = ag_square(tape, diff);

    printf("Modèle : y_hat = x*w + b\n");
    printf("x=2, w=3, b=1, y=5\n");
    printf("y_hat = %.3f\n", ag_data(tape, y_hat));
    printf("perte = %.3f\n", ag_data(tape, loss));

    ag_backward(tape, loss);
    printf("dL/dx = %.3f\n", ag_grad(tape, x));
    printf("dL/dw = %.3f\n", ag_grad(tape, w));
    printf("dL/db = %.3f\n", ag_grad(tape, b));

    ag_tape_free(tape);
}

static void demo_tensor_autograd(void) {
    print_title("Aides autograd tensorielles");

    AgTape* tape = ag_tape_create();

    AgVal x[6];
    AgVal bias[3];
    AgVal shifted[6];

    float x_values[] = {1, 2, 3, 4, 5, 6};
    float bias_values[] = {10, 20, 30};
    for (int i = 0; i < 6; i++) x[i] = ag_leaf(tape, x_values[i]);
    for (int i = 0; i < 3; i++) bias[i] = ag_leaf(tape, bias_values[i]);

    ag_tensor_broadcast_binary_2d(tape, x, 2, 3, bias, 1, 3, shifted, ag_add);

    printf("X + biais :\n");
    for (int i = 0; i < 2; i++) {
        printf("[");
        for (int j = 0; j < 3; j++) {
            if (j > 0) printf(", ");
            printf("%7.3f", ag_data(tape, shifted[i * 3 + j]));
        }
        printf(" ]\n");
    }

    AgVal loss = ag_sum(tape, shifted, 6);
    printf("perte = sum(X + biais) = %.3f\n", ag_data(tape, loss));
    ag_backward(tape, loss);

    printf("Gradients pour le biais :\n");
    for (int j = 0; j < 3; j++) {
        printf("  db[%d] = %.3f\n", j, ag_grad(tape, bias[j]));
    }

    ag_tape_free(tape);
}

static void demo_nn_module(void) {
    print_title("Module nn");

    AgTape* tape = ag_tape_create();
    MtLinear* linear = mt_linear_create(tape, 2, 1, 1);
    MtSequential* model = mt_sequential_create(2);
    if (!tape || !linear || !model) {
        printf("Impossible de créer le modèle nn.\n");
        mt_sequential_free(model);
        mt_linear_free(linear);
        ag_tape_free(tape);
        return;
    }

    mt_linear_set_weight(tape, linear, 0, 0, 0.25f);
    mt_linear_set_weight(tape, linear, 0, 1, -0.50f);
    mt_linear_set_bias(tape, linear, 0, 0.10f);

    mt_sequential_add_linear(model, linear);
    mt_sequential_add_activation(model, MT_ACT_SIGMOID);

    AgVal input[2] = {
        ag_leaf(tape, 1.0f),
        ag_leaf(tape, -2.0f)
    };
    AgVal target[1] = {
        ag_leaf(tape, 1.0f)
    };
    AgVal pred[1];

    if (!mt_sequential_forward(tape, model, input, 2, pred, 1)) {
        printf("Le passage forward du modèle a échoué.\n");
        mt_sequential_free(model);
        mt_linear_free(linear);
        ag_tape_free(tape);
        return;
    }
    AgVal loss = mt_bce_loss(tape, pred, target, 1);

    printf("Modèle : Sequential(Linear(2, 1), Sigmoid)\n");
    printf("Entrée : [1.0, -2.0]\n");
    printf("Cible  : 1.0\n");
    printf("Prédiction = %.4f\n", ag_data(tape, pred[0]));
    printf("Perte BCE  = %.4f\n", ag_data(tape, loss));

    ag_backward(tape, loss);
    printf("Gradients de Linear :\n");
    printf("  dL/dw0 = %.4f\n", ag_grad(tape, mt_linear_weight(linear, 0, 0)));
    printf("  dL/dw1 = %.4f\n", ag_grad(tape, mt_linear_weight(linear, 0, 1)));
    printf("  dL/db  = %.4f\n", ag_grad(tape, mt_linear_bias(linear, 0)));

    mt_sequential_free(model);
    mt_linear_free(linear);
    ag_tape_free(tape);
}

static void demo_optim_module(void) {
    print_title("Module optim");

    AgTape* tape = ag_tape_create();
    MtOptimizer* optim = mt_sgd_create(0.1f);
    if (!tape || !optim) {
        printf("Impossible de créer l'optimiseur.\n");
        mt_optimizer_free(optim);
        ag_tape_free(tape);
        return;
    }

    AgVal w = ag_leaf(tape, 5.0f);
    AgVal target = ag_leaf(tape, 3.0f);
    mt_optimizer_add_param(optim, w);

    AgVal diff = ag_sub(tape, w, target);
    AgVal loss = ag_square(tape, diff);

    printf("Objectif : minimiser (w - 3)^2 avec SGD\n");
    printf("Avant : w = %.3f, perte = %.3f\n", ag_data(tape, w), ag_data(tape, loss));

    mt_optimizer_zero_grad(tape, optim);
    ag_backward(tape, loss);
    printf("Gradient : dL/dw = %.3f\n", ag_grad(tape, w));

    mt_optimizer_step(tape, optim);
    printf("Après un step : w = %.3f\n", ag_data(tape, w));

    mt_optimizer_free(optim);
    ag_tape_free(tape);
}

static void demo_serialization_module(void) {
    print_title("Sauvegarde modèle");

    AgTape* tape = ag_tape_create();
    MtLinear* linear = mt_linear_create(tape, 2, 1, 1);
    if (!tape || !linear) {
        printf("Impossible de créer la couche Linear.\n");
        mt_linear_free(linear);
        ag_tape_free(tape);
        return;
    }

    mt_linear_set_weight(tape, linear, 0, 0, 1.25f);
    mt_linear_set_weight(tape, linear, 0, 1, -0.75f);
    mt_linear_set_bias(tape, linear, 0, 0.50f);

    printf("Avant sauvegarde : w0=%.2f, w1=%.2f, b=%.2f\n",
           ag_data(tape, mt_linear_weight(linear, 0, 0)),
           ag_data(tape, mt_linear_weight(linear, 0, 1)),
           ag_data(tape, mt_linear_bias(linear, 0)));

    if (!mt_save_linear(tape, linear, "linear_demo.mt")) {
        printf("Sauvegarde échouée.\n");
        mt_linear_free(linear);
        ag_tape_free(tape);
        return;
    }

    mt_linear_set_weight(tape, linear, 0, 0, 0.0f);
    mt_linear_set_weight(tape, linear, 0, 1, 0.0f);
    mt_linear_set_bias(tape, linear, 0, 0.0f);

    if (!mt_load_linear(tape, linear, "linear_demo.mt")) {
        printf("Chargement échoué.\n");
        mt_linear_free(linear);
        ag_tape_free(tape);
        return;
    }

    printf("Après chargement : w0=%.2f, w1=%.2f, b=%.2f\n",
           ag_data(tape, mt_linear_weight(linear, 0, 0)),
           ag_data(tape, mt_linear_weight(linear, 0, 1)),
           ag_data(tape, mt_linear_bias(linear, 0)));
    printf("Fichier créé : linear_demo.mt\n");

    mt_linear_free(linear);
    ag_tape_free(tape);
}

static void demo_multiclass_module(void) {
    print_title("Softmax et CrossEntropy");

    AgTape* tape = ag_tape_create();
    if (!tape) {
        printf("Impossible de créer le tape autograd.\n");
        return;
    }

    AgVal logits[3] = {
        ag_leaf(tape, 1.0f),
        ag_leaf(tape, 2.0f),
        ag_leaf(tape, 3.0f)
    };
    AgVal probs[3];
    AgVal target[3] = {
        ag_leaf(tape, 0.0f),
        ag_leaf(tape, 0.0f),
        ag_leaf(tape, 1.0f)
    };

    mt_softmax(tape, logits, 3, probs);
    AgVal loss = mt_cross_entropy_from_logits(tape, logits, target, 3);

    printf("Logits : [1.0, 2.0, 3.0]\n");
    printf("Cible  : classe 2\n");
    printf("Probas : [%.4f, %.4f, %.4f]\n",
           ag_data(tape, probs[0]),
           ag_data(tape, probs[1]),
           ag_data(tape, probs[2]));
    printf("Perte CrossEntropyFromLogits = %.4f\n", ag_data(tape, loss));

    ag_backward(tape, loss);
    printf("Gradient du logit cible = %.4f\n", ag_grad(tape, logits[2]));

    ag_tape_free(tape);
}

static int find_demo_dataset_path(char* out, int out_size) {
    const char* candidates[] = {
        "examples/datasets/dataset_fictif_binaire.csv",
        "../examples/datasets/dataset_fictif_binaire.csv",
        "../../examples/datasets/dataset_fictif_binaire.csv",
        "dataset_fictif_binaire.csv"
    };

    for (int i = 0; i < 4; i++) {
        FILE* file = fopen(candidates[i], "r");
        if (file) {
            fclose(file);
            snprintf(out, (size_t)out_size, "%s", candidates[i]);
            return 1;
        }
    }
    return 0;
}

static int demo_forward_binary(AgTape* tape,
                               const MtModel* model,
                               const AgVal* input,
                               AgVal* prob) {
    return mt_model_forward(tape, model, input, model->input_size, prob, 1);
}

static float demo_dataset_train_batch(AgTape* tape,
                                      MtModel* model,
                                      MtOptimizer* optim,
                                      int graph_checkpoint,
                                      const MtBatch* batch,
                                      int count) {
    ag_rewind(tape, graph_checkpoint);

    AgVal pred[32];
    AgVal target[32];
    AgVal input[2];

    for (int i = 0; i < count; i++) {
        input[0] = ag_leaf(tape, mt_batch_feature(batch, i, 0));
        input[1] = ag_leaf(tape, mt_batch_feature(batch, i, 1));
        if (!demo_forward_binary(tape, model, input, &pred[i])) {
            pred[i] = ag_leaf(tape, 0.5f);
        }
        target[i] = ag_leaf(tape, mt_batch_label(batch, i));
    }

    AgVal loss = mt_bce_loss(tape, pred, target, count);
    float loss_value = ag_data(tape, loss);

    mt_optimizer_zero_grad(tape, optim);
    ag_backward(tape, loss);
    mt_optimizer_step(tape, optim);
    return loss_value;
}

static void demo_dataset_train(AgTape* tape,
                               MtModel* model,
                               MtOptimizer* optim,
                               int graph_checkpoint,
                               MtDataset* train,
                               int epochs,
                               int batch_size) {
    MtBatch* batch = mt_batch_create(batch_size, train->n_features);
    if (!batch) {
        printf("Impossible de créer les mini-batches.\n");
        return;
    }

    printf("\nEntraînement sur train\n");
    printf("Époque   Perte moyenne\n");
    printf("----------------------\n");

    int print_every = epochs / 5;
    if (print_every < 1) print_every = 1;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;
        int seen = 0;
        int n_batches = mt_dataset_num_batches(train, batch_size);

        mt_dataset_shuffle(train);
        for (int b = 0; b < n_batches; b++) {
            int count = mt_dataset_get_batch(train, b, batch_size, batch);
            if (count <= 0) continue;

            float loss = demo_dataset_train_batch(tape, model, optim, graph_checkpoint, batch, count);
            total_loss += loss * (float)count;
            seen += count;
        }

        if (epoch == 1 || epoch % print_every == 0 || epoch == epochs) {
            printf("%-8d %.6f\n", epoch, total_loss / (float)seen);
        }
    }

    mt_batch_free(batch);
}

static float demo_dataset_predict_prob(AgTape* tape,
                                       const MtModel* model,
                                       int graph_checkpoint,
                                       const MtDataset* dataset,
                                       int index) {
    float features[2] = {
        mt_dataset_feature(dataset, index, 0),
        mt_dataset_feature(dataset, index, 1)
    };
    float output[1];

    if (!mt_model_predict(tape, model, graph_checkpoint, features, output, 1)) {
        return 0.5f;
    }
    return output[0];
}

static void print_eval_result(const char* name, const MtEvalResult* result) {
    printf("%s : perte=%.6f, exactitude=%.2f %% (%d/%d)\n",
           name,
           result->loss_mean,
           result->accuracy * 100.0f,
           result->correct,
           result->total);
}

static void demo_fake_dataset_pipeline(void) {
    print_title("Charger un dataset fictif, splitter, choisir un modèle");

    char path[256];
    if (!find_demo_dataset_path(path, (int)sizeof(path))) {
        printf("Dataset introuvable.\n");
        printf("Fichier attendu : examples/datasets/dataset_fictif_binaire.csv\n");
        return;
    }

    printf("Dataset trouvé : %s\n", path);
    printf("Format : x1, x2, y\n");
    printf("Tâche : classification binaire\n");

    MtDataset* dataset = mt_dataset_load_csv(path, 2, 1);
    if (!dataset) {
        printf("Chargement du dataset échoué.\n");
        return;
    }

    printf("\nDataset chargé : %d exemples, %d variables\n", dataset->n_samples, dataset->n_features);

    float train_pct = 70.0f;
    float val_pct = 15.0f;
    printf("\nPourcentage train : ");
    scanf("%f", &train_pct);
    printf("Pourcentage validation : ");
    scanf("%f", &val_pct);

    if (train_pct <= 0.0f || val_pct < 0.0f || train_pct + val_pct >= 100.0f) {
        printf("Pourcentages invalides. Valeurs utilisées : train=70, validation=15, test=15.\n");
        train_pct = 70.0f;
        val_pct = 15.0f;
    }

    int model_choice = 1;
    printf("\nChoisis un modèle\n");
    printf("1. Régression logistique : Linear(2, 1) + Sigmoid\n");
    printf("2. Petit MLP : Linear(2, 4) + Tanh + Linear(4, 1) + Sigmoid\n");
    printf("Choix : ");
    scanf("%d", &model_choice);
    if (model_choice != 1 && model_choice != 2) {
        printf("Choix invalide. Modèle 1 utilisé.\n");
        model_choice = 1;
    }

    int epochs = 120;
    int batch_size = 4;
    float lr = model_choice == 1 ? 0.08f : 0.05f;
    printf("\nÉpoques : ");
    scanf("%d", &epochs);
    if (epochs < 1) epochs = 120;
    printf("Taille de batch : ");
    scanf("%d", &batch_size);
    if (batch_size < 1) batch_size = 4;
    if (batch_size > 32) batch_size = 32;
    printf("Taux d'apprentissage : ");
    scanf("%f", &lr);
    if (lr <= 0.0f) lr = model_choice == 1 ? 0.08f : 0.05f;

    srand(7);
    mt_dataset_shuffle(dataset);

    MtDataset* train = NULL;
    MtDataset* val = NULL;
    MtDataset* test = NULL;
    if (!mt_dataset_split(dataset, train_pct / 100.0f, val_pct / 100.0f, &train, &val, &test)) {
        printf("Split du dataset échoué.\n");
        mt_dataset_free(dataset);
        return;
    }

    printf("Split : train=%d, validation=%d, test=%d\n",
           train->n_samples,
           val ? val->n_samples : 0,
           test->n_samples);

    AgTape* tape = ag_tape_create();
    MtModel* model = NULL;
    MtOptimizer* optim = mt_adam_create(lr, 0.9f, 0.999f, 1e-8f);

    if (model_choice == 1) {
        model = mt_model_create_linear_binary(tape, 2);
    } else {
        model = mt_model_create_mlp_binary(tape, 2, 4);
    }

    if (!tape || !optim || !model) {
        printf("Impossible de créer le modèle.\n");
        mt_optimizer_free(optim);
        mt_model_free(model);
        ag_tape_free(tape);
        mt_dataset_free(train);
        mt_dataset_free(val);
        mt_dataset_free(test);
        mt_dataset_free(dataset);
        return;
    }

    if (model_choice == 1) {
        mt_optimizer_add_linear(optim, model->linear);
        printf("\nModèle choisi : Linear(2, 1) + Sigmoid\n");
    } else {
        mt_optimizer_add_sequential(optim, model->seq);
        printf("\nModèle choisi : petit MLP\n");
    }
    int graph_checkpoint = ag_checkpoint(tape);

    if (model_choice == 1) {
        printf("Poids initiaux : w1=%.4f, w2=%.4f, b=%.4f\n",
               ag_data(tape, mt_linear_weight(model->linear, 0, 0)),
               ag_data(tape, mt_linear_weight(model->linear, 0, 1)),
               ag_data(tape, mt_linear_bias(model->linear, 0)));
    }

    MtLoss loss = mt_loss_create(MT_LOSS_BCE);
    MtTrainConfig train_config = mt_train_config_default();
    train_config.epochs = epochs;
    train_config.batch_size = batch_size;
    train_config.shuffle = 1;

    MtTrainHistory history;
    printf("\nEntraînement sur train\n");
    if (!mt_trainer_train_binary(tape, model, optim, &loss, graph_checkpoint, train, &train_config, &history)) {
        printf("Entraînement échoué.\n");
        mt_optimizer_free(optim);
        mt_model_free(model);
        ag_tape_free(tape);
        mt_dataset_free(train);
        mt_dataset_free(val);
        mt_dataset_free(test);
        mt_dataset_free(dataset);
        return;
    }
    printf("Terminé : %d époques, %d batches, perte finale %.6f\n",
           history.epochs_ran,
           history.batches_seen,
           history.last_loss);

    if (model_choice == 1) {
        printf("\nPoids entraînés : w1=%.4f, w2=%.4f, b=%.4f\n",
               ag_data(tape, mt_linear_weight(model->linear, 0, 0)),
               ag_data(tape, mt_linear_weight(model->linear, 0, 1)),
               ag_data(tape, mt_linear_bias(model->linear, 0)));
    }

    MtEvalResult train_eval;
    MtEvalResult val_eval;
    MtEvalResult test_eval;
    printf("\nÉvaluation\n");
    if (mt_model_eval_binary(tape, model, graph_checkpoint, train, 0.5f, &train_eval)) {
        print_eval_result("Train", &train_eval);
    }
    if (val && mt_model_eval_binary(tape, model, graph_checkpoint, val, 0.5f, &val_eval)) {
        print_eval_result("Validation", &val_eval);
    }
    if (mt_model_eval_binary(tape, model, graph_checkpoint, test, 0.5f, &test_eval)) {
        print_eval_result("Test", &test_eval);
    }

    printf("\nPrédictions sur test\n");
    printf("x1      x2      y vrai   proba    classe\n");
    printf("----------------------------------------\n");
    for (int i = 0; i < test->n_samples; i++) {
        float prob = demo_dataset_predict_prob(tape, model, graph_checkpoint, test, i);
        int pred = prob >= 0.5f ? 1 : 0;
        printf("%-7.2f %-7.2f %-8.0f %-8.4f %d\n",
               mt_dataset_feature(test, i, 0),
               mt_dataset_feature(test, i, 1),
               mt_dataset_label(test, i),
               prob,
               pred);
    }

    printf("\nDataset utilisé : %s\n", path);
    if (mt_model_save(tape, model, "dataset_pipeline_model.mt")) {
        printf("Modèle sauvegardé : dataset_pipeline_model.mt\n");
    } else {
        printf("Sauvegarde du modèle échouée.\n");
    }

    mt_optimizer_free(optim);
    mt_model_free(model);
    ag_tape_free(tape);
    mt_dataset_free(train);
    mt_dataset_free(val);
    mt_dataset_free(test);
    mt_dataset_free(dataset);
}

static void print_menu(void) {
    printf("\nMenu de démo MiniTorch\n");
    printf("1. Bases des tenseurs\n");
    printf("2. Broadcast et réduction\n");
    printf("3. Algèbre linéaire\n");
    printf("4. Chaîne autograd scalaire\n");
    printf("5. Aides autograd tensorielles\n");
    printf("6. Module nn\n");
    printf("7. Module optim\n");
    printf("8. Sauvegarde modèle\n");
    printf("9. Softmax et CrossEntropy\n");
    printf("10. Pipeline dataset fictif\n");
    printf("0. Quitter\n");
    printf("Choix : ");
}

int main(void) {
    configure_console();
    printf("Démo MiniTorch\n");

    while (1) {
        int choice = -1;
        print_menu();
        if (scanf("%d", &choice) != 1) {
            return 1;
        }

        switch (choice) {
            case 1: demo_tensor_basics(); break;
            case 2: demo_broadcast_and_reduce(); break;
            case 3: demo_linalg(); break;
            case 4: demo_scalar_autograd(); break;
            case 5: demo_tensor_autograd(); break;
            case 6: demo_nn_module(); break;
            case 7: demo_optim_module(); break;
            case 8: demo_serialization_module(); break;
            case 9: demo_multiclass_module(); break;
            case 10: demo_fake_dataset_pipeline(); break;
            case 0:
                printf("Au revoir\n");
                return 0;
            default:
                printf("Choix invalide\n");
                break;
        }
    }
}
