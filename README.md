# MiniTorch

MiniTorch est un petit framework de deep learning écrit en C.

Le but du projet est pédagogique. Il permet de comprendre les briques principales d'un framework comme PyTorch, mais avec une base plus simple:

- tenseurs
- autograd
- couches de réseau de neurones
- optimiseurs
- datasets et mini-batches
- sauvegarde et chargement des poids
- exemples d'entraînement

Le projet n'est pas encore une librairie de production. C'est une base d'apprentissage qui grandit étape par étape.

## Fonctionnalités

### Tenseurs

Le module `core/tensor` fournit:

- création de tenseurs
- accès aux valeurs
- reshape
- broadcasting simple
- opérations élémentaires
- somme et moyenne
- produit matriciel
- transpose
- déterminant simple

Fichiers principaux:

```text
include/minitorch/core/tensor.h
include/minitorch/core/tensor_ops.h
include/minitorch/core/tensor_linalg.h
src/core/tensor.c
src/core/tensor_ops.c
src/core/tensor_linalg.c
```

### Autograd

Le module `core/autograd` permet de créer un graphe de calcul et de calculer les gradients avec `backward`.

Il contient aussi un système de checkpoint:

```c
int checkpoint = ag_checkpoint(tape);
ag_rewind(tape, checkpoint);
```

Ce mécanisme garde les paramètres du modèle et supprime le graphe temporaire créé pendant un batch.

Fonctions utiles:

```c
AgTape* tape = ag_tape_create();
AgVal x = ag_leaf(tape, 2.0f);
AgVal y = ag_leaf(tape, 3.0f);
AgVal z = ag_mul(tape, x, y);
ag_backward(tape, z);
printf("%f\n", ag_grad(tape, x));
ag_tape_free(tape);
```

### Neural network

Le module `nn` ajoute des briques de réseau:

- `MtLinear`
- `MtSequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `Softmax`
- `MSELoss`
- `BCELoss`
- `CrossEntropyLoss`
- `CrossEntropyFromLogits`

Exemple simple:

```c
AgTape* tape = ag_tape_create();
MtLinear* layer = mt_linear_create(tape, 2, 1, 1);

AgVal input[2] = {
    ag_leaf(tape, 1.0f),
    ag_leaf(tape, 2.0f)
};
AgVal output[1];

mt_linear_forward(tape, layer, input, output);
```

### Optimiseurs

Le module `optim` contient:

- SGD
- Momentum
- Adam

Exemple:

```c
MtOptimizer* optim = mt_adam_create(0.01f, 0.9f, 0.999f, 1e-8f);
mt_optimizer_add_linear(optim, layer);

mt_optimizer_zero_grad(tape, optim);
ag_backward(tape, loss);
mt_optimizer_step(tape, optim);
```

### Data

Le module `data` permet de travailler avec:

- `MtDataset`
- `MtBatch`
- mini-batches
- shuffle

Exemple:

```c
MtDataset* dataset = mt_dataset_create(x, y, n_samples, n_features);
MtBatch* batch = mt_batch_create(batch_size, n_features);

mt_dataset_shuffle(dataset);
int count = mt_dataset_get_batch(dataset, 0, batch_size, batch);
```

### Sauvegarde des poids

Le module `serialization` permet de sauvegarder et recharger les poids:

```c
mt_save_linear(tape, layer, "model.mt");
mt_load_linear(tape, layer, "model.mt");

mt_save_sequential(tape, model, "model.mt");
mt_load_sequential(tape, model, "model.mt");
```

Le format est en texte simple. Il peut être ouvert et lu facilement.

## Compilation

Le projet utilise CMake.

Sous Linux ou WSL:

```bash
mkdir -p build
cd build
cmake ..
make
```

Lancer tous les tests:

```bash
ctest --output-on-failure
```

## Programmes disponibles

Après compilation, les exécutables principaux sont:

```text
example
demo_train
demo_xor
demo_multiclass
test_tensor
test_autograd
test_nn
test_optim
test_data
test_xor
test_serialization
test_multiclass
```

### `example`

Menu interactif qui montre les bases du projet:

- tenseurs
- broadcasting
- algèbre linéaire
- autograd scalaire
- helpers autograd tensoriels
- module `nn`
- module `optim`
- sauvegarde de modèle
- softmax et cross-entropy

Lancement:

```bash
./example
```

### `demo_train`

Démo de régression logistique binaire.

Elle utilise:

- `MtDataset`
- mini-batches
- `Linear`
- `Sigmoid`
- `BCELoss`
- `Adam`
- checkpoint et rewind du tape

Lancement:

```bash
./demo_train
```

### `demo_xor`

Démo d'un MLP qui apprend XOR.

Architecture:

```text
Linear(2, 4)
Tanh
Linear(4, 1)
Sigmoid
```

La démo sauvegarde le modèle entraîné dans:

```text
xor_model.mt
```

Lancement:

```bash
./demo_xor
```

### `demo_multiclass`

Démo de classification multi-classe.

Architecture:

```text
Linear(2, 3)
CrossEntropyFromLogits
```

`Softmax` reste utilisé pour afficher les probabilités finales.

Lancement:

```bash
./demo_multiclass
```

## Exemple de boucle d'entraînement

Voici la forme générale d'une boucle d'entraînement dans MiniTorch:

```c
AgTape* tape = ag_tape_create();

MtLinear* model = mt_linear_create(tape, n_features, 1, 1);
MtOptimizer* optim = mt_adam_create(0.01f, 0.9f, 0.999f, 1e-8f);
mt_optimizer_add_linear(optim, model);

int checkpoint = ag_checkpoint(tape);

for (int epoch = 0; epoch < epochs; epoch++) {
    mt_dataset_shuffle(dataset);

    for (int batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        ag_rewind(tape, checkpoint);

        /* forward */
        /* loss */

        mt_optimizer_zero_grad(tape, optim);
        ag_backward(tape, loss);
        mt_optimizer_step(tape, optim);
    }
}
```

L'idée importante:

- les poids restent dans le tape
- le graphe du batch est temporaire
- `ag_rewind` supprime ce graphe temporaire

## Structure du projet

```text
include/minitorch/
  core/
  nn/
  optim/
  data/
  serialization/

src/
  core/
  nn/
  optim/
  data/
  serialization/

examples/
  main.c
  demo_train.c
  demo_xor.c
  demo_multiclass.c

tests/
  test_tensor.c
  test_autograd.c
  test_nn.c
  test_optim.c
  test_data.c
  test_xor.c
  test_serialization.c
  test_multiclass.c
```

## Limites actuelles

MiniTorch reste volontairement simple. Il manque encore plusieurs éléments importants:

- autograd tensoriel complet
- broadcasting général dans tous les gradients
- softmax avec stabilité numérique renforcée pour les très grands logits
- vraie API de module plus proche de PyTorch
- initialisations Xavier et He
- séparation train/test
- chargement de datasets depuis fichiers
- sauvegarde plus robuste avec versionnement plus strict
- gestion mémoire plus avancée

## Roadmap

Les prochaines étapes possibles:

1. Ajouter des initialisations de poids propres:
   - Xavier
   - He

2. Ajouter un vrai split de dataset:
   - train
   - validation
   - test

3. Ajouter des métriques:
   - accuracy
   - loss moyenne
   - matrice de confusion simple

4. Ajouter une API d'évaluation:
   - évaluer un modèle sans entraîner
   - afficher les prédictions
   - comparer train et validation

5. Nettoyer l'API publique:
   - noms plus cohérents
   - erreurs mieux signalées
   - documentation par fonction

6. Rendre l'autograd plus tensoriel:
   - gradients de `matmul`
   - gradients de `sum`
   - gradients de broadcasting
   - vues et reshape

## Objectif du projet

MiniTorch sert à apprendre comment fonctionne un petit framework de deep learning.

Il ne cherche pas à remplacer PyTorch. Il sert plutôt à comprendre, en C, les idées derrière:

- forward
- loss
- backward
- gradients
- optimiseur
- mini-batches
- modèles multi-couches
- classification binaire et multi-classe
