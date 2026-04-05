import os

import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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


# 2. Data Loading & Merging
base = os.path.join(os.path.dirname(__file__), "..", "..")
train_df = pd.read_csv(os.path.join(base, "data", "task2", "train_df.csv"))
val_df = pd.read_csv(os.path.join(base, "data", "task2", "val_df.csv"))
test_df = pd.read_csv(os.path.join(base, "data", "task2", "dev_df.csv"))

train_df = pd.concat([train_df, val_df], ignore_index=True)
for df in (train_df, test_df):
    df["clean_lyrics"] = df["lyrics"].apply(preprocess_lyrics)

# Define label columns
label_cols = ["sexualization", "violence", "hate"]

# Build multilabel targets as numpy arrays (shape: n_samples x 3)
Y = train_df[label_cols].fillna(0).astype(int).values
Y_test = test_df[label_cols].fillna(0).astype(int).values

# 3. Vectorization (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(train_df["clean_lyrics"])
X_test = tfidf.transform(test_df["clean_lyrics"])

# 4. Train / validation split
# NOTE: stratifying a true multilabel y with sklearn's train_test_split isn't supported.
# If you need stratified multilabel splits, use iterative stratification packages.
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. Define model candidates (wrapped in OneVsRestClassifier for multilabel)
candidates = [
    {
        "name": "LogisticRegression_OVR",
        "estimator": OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "param_grid": {
            # GridSearch will pass params to the underlying estimator via 'estimator__estimator__C'
            "estimator__C": [0.1, 1.0, 10.0]
        },
    },
    {
        "name": "RandomForest_OVR",
        "estimator": OneVsRestClassifier(
            RandomForestClassifier(class_weight="balanced", n_jobs=-1)
        ),
        "param_grid": {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [None, 20],
        },
    },
    {
        "name": "LinearSVC_OVR",
        "estimator": OneVsRestClassifier(LinearSVC(max_iter=20000)),
        "param_grid": {"estimator__C": [0.01, 0.1, 1.0]},
    },
]

# 6. Training + evaluation loop
results = []
for c in candidates:
    name = c["name"]
    print(f"\n--- Training candidate: {name} ---")

    # Build pipeline so GridSearch can tune inside the OVR wrapper
    pipe = Pipeline([("clf", c["estimator"])])

    # Grid search (use a small CV to keep runtime reasonable)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid={"clf__" + k: v for k, v in c["param_grid"].items()},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print(f"Best params for {name}: {gs.best_params_}")

    # Predict on test set
    preds = best.predict(X_test)

    # Per-label reports
    report_per_label = {}
    for i, col in enumerate(label_cols):
        rep = classification_report(
            Y_test[:, i], preds[:, i], output_dict=True, zero_division=0
        )
        report_per_label[col] = rep

    # Aggregated multilabel metrics
    f1_micro = f1_score(Y_test, preds, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, preds, average="macro", zero_division=0)
    ham_loss = hamming_loss(Y_test, preds)

    results.append(
        {
            "model": name,
            "best_params": gs.best_params_,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "hamming_loss": ham_loss,
            "per_label_reports": report_per_label,
        }
    )

    # Print readable results
    print(
        f"\nResults for {name}: f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}, hamming_loss={ham_loss:.4f}"
    )
    for col in label_cols:
        print(f"\n--- Label: {col} ---")
        print(
            classification_report(
                Y_test[:, label_cols.index(col)],
                preds[:, label_cols.index(col)],
                zero_division=0,
            )
        )

# 7. Summarize
print("\n=== SUMMARY ===")
for r in results:
    print(
        f"{r['model']}: f1_micro={r['f1_micro']:.4f}, f1_macro={r['f1_macro']:.4f}, hamming={r['hamming_loss']:.4f}"
    )
