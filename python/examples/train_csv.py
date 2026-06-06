from __future__ import annotations

import argparse
from pathlib import Path

import minitorch as mt


def build_model(dataset: mt.Dataset, hidden_size: int, model_type: str):
    output_size = 1 if dataset.is_binary else dataset.n_classes
    if model_type == "linear":
        return mt.Linear(input_size=dataset.n_features, output_size=output_size)
    return mt.MLP(input_size=dataset.n_features, hidden_size=hidden_size, output_size=output_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîner un modèle MiniTorch depuis Python.")
    parser.add_argument("csv", type=Path, help="Chemin du fichier CSV.")
    parser.add_argument("--no-header", action="store_true", help="Indique que le CSV n'a pas d'en-tête.")
    parser.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--save", type=Path, default=Path("python_model.mt"))
    args = parser.parse_args()

    data = mt.Dataset.from_csv(args.csv, has_header=not args.no_header)
    print(f"Dataset: {data.n_samples} lignes, {data.n_features} features, {data.n_classes} classes")

    model = build_model(data, hidden_size=args.hidden_size, model_type=args.model)
    trainer = mt.Trainer(model, optimizer="adam", lr=args.lr, loss="auto")
    result = trainer.fit(data, epochs=args.epochs, batch_size=args.batch_size)

    print("Historique:", result.history)
    print("Train:", result.train)
    print("Validation:", result.validation)
    print("Test:", result.test)

    first_features, first_label = data.head(1)[0]
    print("Première prédiction:", model.predict(first_features), "label:", first_label)

    model.save(args.save)
    print(f"Modèle sauvegardé: {args.save}")

    model.close()
    data.close()


if __name__ == "__main__":
    main()
