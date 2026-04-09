import argparse
import os
import sys

import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import DEFAULT_SEED, set_seed

set_seed(DEFAULT_SEED)

# 1. Load Spacy for Spanish Preprocessing
nlp = spacy.load("es_core_news_sm")


def preprocess_lyrics(text):
    doc = nlp(str(text).lower())
    # Tokenize, remove stopwords/punctuation, and lemmatize
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return " ".join(tokens)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="ML Baseline Comparison for Task 1"
    )
    arg_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Whether to use original dataset or not (default: False)",
    )
    args = arg_parser.parse_args()
    pre = "processed_" if not args.baseline else ""
    # 2. Data Loading & Merging
    print(f"Loading data for Task 1 with baseline={args.baseline}...")
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    train_df = pd.read_csv(os.path.join(base, "data", "task1", f"{pre}train_df.csv"))
    val_df = pd.read_csv(os.path.join(base, "data", "task1", f"{pre}val_df.csv"))
    test_df = pd.read_csv(os.path.join(base, "data", "task1", f"{pre}dev_df.csv"))

    train_df = pd.concat([train_df, val_df], ignore_index=True)
    print("Sample:", train_df[["lyrics", "label"]].head(2))
    print("Preprocessing text...")
    train_df["clean_lyrics"] = train_df["lyrics"].apply(preprocess_lyrics)
    test_df["clean_lyrics"] = test_df["lyrics"].apply(preprocess_lyrics)

    label_map = {"NM": 0, "M": 1}
    y_train = train_df["label"].map(label_map).values
    y_test = test_df["label"].map(label_map).values

    # 3. Vectorization (TF-IDF)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df["clean_lyrics"])
    X_test = tfidf.transform(test_df["clean_lyrics"])

    # 4. Model Definitions & Parameter Spaces
    models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(max_iter=1000),
            "param_grid": {
                "C": Continuous(1e-3, 1e2, distribution="log-uniform"),
                "class_weight": Categorical(["balanced", None]),
            },
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(class_weight="balanced"),
            "param_grid": {
                "n_estimators": Integer(50, 500),
                "max_depth": Integer(5, 50),
                "min_samples_split": Integer(2, 11),
            },
        },
        {
            "name": "SVM",
            "estimator": SVC(class_weight="balanced"),
            "param_grid": {
                "C": Continuous(1e-3, 1e2, distribution="log-uniform"),
                "kernel": Categorical(["linear", "rbf"]),
            },
        },
    ]

    # 5. Training with GASearchCV
    results_list = []
    for m in models:
        print(f"\n--- Optimizing {m['name']} ---")
        gs = GASearchCV(
            estimator=m["estimator"],
            cv=3,
            param_grid=m["param_grid"],
            scoring="f1_macro",
            n_jobs=-1,
            population_size=20,
            generations=10,
            verbose=True,
        )
        gs.fit(X_train, y_train)

        # Evaluation
        preds = gs.predict(X_test)
        print(f"Best Params: {gs.best_params_}")
        print(classification_report(y_test, preds))

        precision, recall, f1_opt, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0.0
        )
        acc_opt = accuracy_score(y_test, preds)

        results_list.append(
            {
                "Modelo": m["name"],
                "F1-Macro": f1_opt,
                "Accuracy": acc_opt,
                "Precision": precision,
                "Recall": recall,
                "Best Params": str(gs.best_params_),
            }
        )

    # 6. Save Results
    is_baseline_str = "baseline" if args.baseline else "processed"
    results_file = os.path.join(
        base, "results", "task1", f"ml_{is_baseline_str}_results.csv"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df_res = pd.DataFrame(results_list).sort_values("F1-Macro", ascending=False)
    df_res.to_csv(results_file, index=False)

    print(f"\n{'=' * 60}")
    print("RESULTADOS COMPARATIVA")
    print(f"{'=' * 60}")
    print(df_res.to_markdown(index=False))
    print(f"\nGuardado en: {results_file}")
