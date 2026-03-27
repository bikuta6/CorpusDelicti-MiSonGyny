import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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


# 2. Data Loading & Merging
# Assuming 'train.csv' exists with columns: song_id, lyrics, label
train_df = pd.read_csv("../../data/task3/train_df.csv")
val_df = pd.read_csv("../../data/task3/val_df.csv")
train_df = pd.concat([train_df, val_df], ignore_index=True)
train_df["clean_lyrics"] = train_df["lyrics"].apply(preprocess_lyrics)

# Test data (separate files)
test_df = pd.read_csv("../../data/task3/dev_df.csv")  # song_id, lyrics
test_df["clean_lyrics"] = test_df["lyrics"].apply(preprocess_lyrics)
# Create a mapping
label_map = {"N": 0, "Y": 1}

# Apply to training data
y = train_df["label"].map(label_map)

# Apply to test data
y_test = test_df["label"].map(label_map)

# 3. Vectorization (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_df["clean_lyrics"])
y_train = y

X_test = tfidf.transform(test_df["clean_lyrics"])

# 4. Model Definitions & Genetic Parameter Spaces
models = [
    {
        "name": "RandomForest",
        "estimator": RandomForestClassifier(),
        "params": {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(5, 30),
            "min_samples_split": Integer(2, 10),
        },
    },
    {
        "name": "SVM",
        "estimator": SVC(),
        "params": {
            "C": Continuous(0.1, 10, distribution="log-uniform"),
            "kernel": Categorical(["linear", "rbf"]),
        },
    },
    {
        "name": "LogisticRegression",
        "estimator": LogisticRegression(max_iter=1000),
        "params": {
            "C": Continuous(0.1, 10, distribution="log-uniform"),
        },
    },
]

# 5. Training with Genetic Algorithms
for m in models:
    print(f"--- Optimizing {m['name']} ---")
    evolve = GASearchCV(
        estimator=m["estimator"],
        cv=3,
        scoring="f1_macro",
        param_grid=m["params"],
        population_size=10,
        generations=5,
    )

    evolve.fit(X_train, y_train)

    # Evaluation
    preds = evolve.predict(X_test)
    print(f"Best Params: {evolve.best_params_}")
    print(classification_report(y_test, preds))
