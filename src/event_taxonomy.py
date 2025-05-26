import argparse
import sys
from functools import lru_cache
from pathlib import Path

import fasttext
import hdbscan
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cc.en.300.bin"

if not MODEL_PATH.exists():
    sys.exit(
        f"FastText model missing at {MODEL_PATH}.  See README for download cmd."
    )

_ft_model = fasttext.load_model(str(MODEL_PATH))


@lru_cache(maxsize=None)
def _sent_vec(text: str) -> np.ndarray:
    return _ft_model.get_sentence_vector(text)


def load_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["event_ts"]).assign(
        event_name=lambda d: d["event_name"].str.strip()
    )
    assert {"profile_id", "event_ts", "event_name", "purchased_7d"}.issubset(
        df
    ), "Missing required columns"
    return df


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    vecs = np.vstack(df["event_name"].apply(_sent_vec).to_numpy())
    df["vec"] = list(vecs)

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean", min_cluster_size=5, min_samples=2
    ).fit(vecs)

    labels = clusterer.labels_
    noise_fix = np.where(labels == -1, df.index + 10_000, labels)
    df["canonical_event"] = noise_fix.astype(int)

    df["_tmp"] = 1
    alias = (
        df.groupby("canonical_event")[["event_name", "_tmp"]]
        .apply(lambda g: g["event_name"].mode().iloc[0])
        .rename("canonical_alias")
    )
    df = df.join(alias, on="canonical_event")
    return df


@lru_cache(maxsize=1)
def _load_action_clf():
    from joblib import load

    model_path = Path(__file__).parents[1] / "models" / "event_action_clf.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run "
            "`python -m src.action_classifier train ...` first."
        )
    return load(model_path)


def is_brand(text: str) -> bool:
    clf = _load_action_clf()
    return clf.predict([text])[0] == "BRAND"


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["profile_id", "event_ts"])

    df["n_events_24h"] = df.groupby("profile_id")["event_ts"].transform(
        lambda s: s.diff().dt.total_seconds().fillna(0).lt(86_400).cumsum()
    )

    df["is_brand_action"] = df["event_name"].apply(is_brand).astype(int)

    X = pd.get_dummies(df["canonical_event"].astype(str), prefix="ev").join(
        df[["n_events_24h", "is_brand_action"]]
    )
    y = df["purchased_7d"].astype(int)
    return X, y


def train_value_model(X: pd.DataFrame, y: pd.Series):
    dtrain = lgb.Dataset(X, label=y)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=30,
        feature_fraction=0.8,
        metric="auc",
        verbosity=-1,
    )

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=200,
        valid_sets=[dtrain],
        valid_names=["train"],
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    shap_df = (
        pd.DataFrame(
            {"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )
    return model, shap_df


def _cli(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    ns = p.parse_args(args)

    df = load_events(Path(ns.csv))
    df = canonicalise(df)
    X, y = build_feature_matrix(df)
    _, shap_top = train_value_model(X, y)

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    alias_map = {
        f"ev_{cid}": alias
        for cid, alias in df.drop_duplicates("canonical_event")
        .set_index("canonical_event")["canonical_alias"]
        .items()
    }

    def rename(f):
        if f.startswith("ev_"):
            return alias_map.get(f, f)
        if f == "n_events_24h":
            return "events_24h_count"
        if f == "is_brand_action":
            return "is_brand_action (1 = brand)"
        return f

    shap_top["feature"] = shap_top["feature"].map(rename)
    print("\nTop predictors (mean |SHAP|)\n")
    print(shap_top.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    _cli()
