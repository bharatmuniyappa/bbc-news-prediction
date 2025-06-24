from __future__ import annotations

import joblib
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class BBCNewsClassifier:
    def __init__(
        self,
        *,
        ngrams: tuple[int, int] = (1, 2),
        max_df: float = 0.9,
        C: float = 4.0,
        max_iter: int = 250,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline(
            steps=[
                (
                    "vectoriser",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=ngrams,
                        max_df=max_df,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(
        self,
        texts: pd.Series,
        labels: pd.Series,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self) -> dict[str, float]:
        preds = self.pipeline.predict(self.X_test)
        report = classification_report(self.y_test, preds, output_dict=True, zero_division=0)
        holdout_acc = accuracy_score(self.y_test, preds)
        cv_acc = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=5, n_jobs=-1).mean()

        return {
            "holdout_accuracy": holdout_acc,
            "cross_val_accuracy": cv_acc,
            "per_class_f1": {
                label: metrics["f1-score"]
                for label, metrics in report.items()
                if label not in {"accuracy", "macro avg", "weighted avg"}
            },
        }

    def plot_confusion_matrix(self) -> None:
        ConfusionMatrixDisplay.from_estimator(self.pipeline, self.X_test, self.y_test, xticks_rotation=45)
        plt.tight_layout()
        plt.show()

    def save(self, path: pathlib.Path | str = "bbc_pipeline.joblib") -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: pathlib.Path | str):
        obj = cls.__new__(cls)
        obj.pipeline = joblib.load(path)
        return obj
