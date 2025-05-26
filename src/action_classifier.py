from __future__ import annotations
import argparse
from pathlib import Path
import joblib, pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

KEYWORDS_BRAND = re.compile(r"(sent|delivered|shipped|refunded)", re.I)

def _norm(series: pd.Series) -> pd.Series:
    return (
        series.str.strip()
              .str.rstrip('.')
              .str.casefold()
    )

def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "label" not in df.columns and "cls" in df.columns:
        df = df.rename(columns={"cls": "label"})
    if {"event_name", "label"} - set(df.columns):
        raise ValueError("CSV must contain event_name and label/cls")

    df["event_name"] = _norm(df["event_name"])
    df["label"] = df["label"].str.upper().str.strip()
    if not df["label"].isin({"CUSTOMER", "BRAND"}).all():
        raise ValueError("Label values must be CUSTOMER or BRAND")
    return df[["event_name", "label"]].drop_duplicates()

def _make_pipeline() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )

def train(labels_csv: Path, out_model: Path) -> Pipeline:
    df = load_labels(labels_csv)
    clf = _make_pipeline().fit(df["event_name"], df["label"])
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_model)
    print(f"✓ Saved classifier → {out_model}")
    return clf

def predict(model: Path | str | Pipeline, text: str) -> str:
    txt_norm = _norm(pd.Series([text]))[0]

    if isinstance(model, (Path, str)):
        model = joblib.load(model)

    proba = model.predict_proba([txt_norm])[0]
    best  = proba.max()
    if best < 0.6:
        return "BRAND" if KEYWORDS_BRAND.search(txt_norm) else "CUSTOMER"
    return model.classes_[proba.argmax()]

def _cli(argv=None):
    ap = argparse.ArgumentParser("Customer-vs-Brand event classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--labels", required=True)
    p_train.add_argument("--model",  required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model",  required=True)
    p_pred.add_argument("text", help="event name")

    ns = ap.parse_args(argv)
    if ns.cmd == "train":
        train(Path(ns.labels), Path(ns.model))
    else:
        print(predict(Path(ns.model), ns.text))

if __name__ == "__main__":
    _cli()
