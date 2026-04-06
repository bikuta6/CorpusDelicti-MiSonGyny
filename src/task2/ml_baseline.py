import os

import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer

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
    # 2. Data Loading & Merging
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    train_df = pd.read_csv(os.path.join(base, "data", "task2", "train_df.csv"))
    val_df = pd.read_csv(os.path.join(base, "data", "task2", "val_df.csv"))
    test_df = pd.read_csv(os.path.join(base, "data", "task2", "dev_df.csv"))

    train_df = pd.concat([train_df, val_df], ignore_index=True)

    print("Preprocessing text...")
    train_df["clean_lyrics"] = train_df["lyrics"].apply(preprocess_lyrics)
    test_df["clean_lyrics"] = test_df["lyrics"].apply(preprocess_lyrics)

    label_cols = ["sexualization", "violence", "hate"]

    # Build multilabel targets as numpy arrays
    y_train = train_df[label_cols].fillna(0).astype(int).values
    y_test = test_df[label_cols].fillna(0).astype(int).values

    # 3. Vectorization (TF-IDF)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df["clean_lyrics"])
    X_test = tfidf.transform(test_df["clean_lyrics"])

    # 4. Model Definitions & Parameter Spaces
    models = [
        {
            "name": "LogisticRegression_OVR",
            "estimator": OneVsRestClassifier(
                LogisticRegression(max_iter=1000, class_weight="balanced")
            ),
            "param_grid": {
                "estimator__C": Continuous(1e-3, 1e2, distribution="log-uniform")
            },
        },
        {
            "name": "RandomForest_OVR",
            "estimator": OneVsRestClassifier(
                RandomForestClassifier(class_weight="balanced", n_jobs=-1)
            ),
            "param_grid": {
                "estimator__n_estimators": Integer(50, 500),
                "estimator__max_depth": Integer(5, 50),
                "estimator__min_samples_split": Integer(2, 11),
            },
        },
        {
            "name": "SVM_OVR",
            "estimator": OneVsRestClassifier(SVC(class_weight="balanced")),
            "param_grid": {
                "estimator__C": Continuous(1e-3, 1e2, distribution="log-uniform"),
                "estimator__kernel": Categorical(["linear", "rbf"]),
            },
        },
    ]

    # 5. Training with GASearchCV
    results = []
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

        f1_micro = f1_score(y_test, preds, average="micro", zero_division=0)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
        ham_loss = hamming_loss(y_test, preds)

        results.append(
            {
                "Modelo": m["name"],
                "F1-Macro": f1_macro,
                "F1-Micro": f1_micro,
                "Hamming Loss": ham_loss,
                "Best Params": str(gs.best_params_),
            }
        )

        print(
            f"f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}, hamming_loss={ham_loss:.4f}"
        )
        for i, col in enumerate(label_cols):
            print(f"\n--- Label: {col} ---")
            print(classification_report(y_test[:, i], preds[:, i], zero_division=0))

    # 6. Save Results
    results_file = os.path.join(base, "results", "task2", "ml_baseline_results.csv")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df_res = pd.DataFrame(results).sort_values("F1-Macro", ascending=False)
    df_res.to_csv(results_file, index=False)

    print(f"\n{'=' * 60}")
    print("RESULTADOS COMPARATIVA")
    print(f"{'=' * 60}")
    print(df_res.to_markdown(index=False))
    print(f"\nGuardado en: {results_file}")
