from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from bot_filter import is_bot

warnings.filterwarnings("ignore", category=UserWarning)

UNSUB_NAMES = {
    "Unsubscribed",
    "Clicked email to unsubscribe",
    "Marked Spam",
    "Spam Complaint",
}
OPEN_RE = r"(?:open|click)"


def _prep_survival(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    df["is_unsub"] = df["event_name"].isin(UNSUB_NAMES)

    first_unsub = (
        df.loc[df.is_unsub]
        .groupby("profile_id")["event_ts"]
        .min()
        .rename("unsub_ts")
    )

    base = (
        df.groupby("profile_id")["event_ts"]
        .min()
        .rename("first_seen_ts")
        .to_frame()
        .join(first_unsub, how="left")
    )

    censor_ts = df["event_ts"].max()
    base["event"] = ~base["unsub_ts"].isna()
    base["duration"] = np.where(
        base["event"],
        (base["unsub_ts"] - base["first_seen_ts"]).dt.days + 1,
        (censor_ts - base["first_seen_ts"]).dt.days + 1,
    )
    return base.reset_index(), censor_ts


def _burst_count(ts: pd.Series) -> int:
    t = ts.sort_values().to_numpy()
    max_cnt = i = 0
    for j in range(len(t)):
        while t[j] - t[i] > np.timedelta64(1, "D"):
            i += 1
        max_cnt = max(max_cnt, j - i + 1)
    return int(max_cnt)


def _opens_last30(group: pd.DataFrame, ref_ts: pd.Timestamp) -> int:
    return (group.loc[group["event_ts"] >= ref_ts, "is_open"]).sum()


def _feature_aggregate(df: pd.DataFrame, censor_ts: pd.Timestamp) -> pd.DataFrame:
    df["is_open"] = df["event_name"].str.contains(OPEN_RE, case=False, regex=True)
    df["is_click"] = df["event_name"].str.contains("click", case=False, regex=True)
    thirty_cut = censor_ts - pd.Timedelta(days=30)

    feat = (
        df.groupby("profile_id", group_keys=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_events_24h": _burst_count(g["event_ts"]),
                    "pct_brand_actions": g["event_name"].isin(UNSUB_NAMES).mean(),
                    "days_since_last_event": (censor_ts - g["event_ts"].max()).days,
                    "opens_last30d": _opens_last30(g, ref_ts=thirty_cut),
                    "is_bot": is_bot(
                        open_count=int(g["is_open"].sum()),
                        click_count=int(g["is_click"].sum()),
                        email=""
                    ),
                }
            )
        )
        .reset_index()
    )

    feat["days_since_last_event"] = feat["days_since_last_event"].clip(0, 60)
    return feat


_TIMESTAMP_ALIASES = {"event_time", "timestamp", "time", "ts"}


def _read_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "event_ts" not in df.columns:
        alias = next((c for c in df.columns if c.lower() in _TIMESTAMP_ALIASES), None)
        if alias is None:
            raise KeyError(
                "CSV must contain 'event_ts' or an alias "
                f"{', '.join(_TIMESTAMP_ALIASES)}"
            )
        df.rename(columns={alias: "event_ts"}, inplace=True)
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
    if df["event_ts"].isna().all():
        raise ValueError("No valid timestamps after parsing 'event_ts'.")
    return df


def load_training_frame(csv_path: Path) -> pd.DataFrame:
    events = _read_events(csv_path)
    surv, censor_ts = _prep_survival(events)
    feat = _feature_aggregate(events, censor_ts)
    return surv.merge(feat, on="profile_id", how="left").fillna(0)


def train_guardrail(csv_path: Path, model_out: Path) -> None:
    df = load_training_frame(csv_path)

    X = df[
        ["n_events_24h", "pct_brand_actions", "days_since_last_event", "opens_last30d", "is_bot"]
    ].to_numpy()
    y = Surv.from_arrays(event=df["event"].to_numpy(), time=df["duration"].to_numpy())

    rsf = RandomSurvivalForest(
        n_estimators=500,
        max_features=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ).fit(X, y)

    c_idx, *_ = concordance_index_censored(
        y["event"], y["time"], -rsf.predict(X)
    )
    print(f"Training concordance-index ≈ {c_idx:0.3f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rsf, model_out)
    print(f"✓ guardrail model saved → {model_out}")


def _risk_to_cooldown(risk7: float, thresh: float = 0.02) -> int:
    if risk7 < thresh:
        return 0
    return int(min(30, (risk7 / thresh) * 7))


def predict_cooldown(model_pkl: Path, **features) -> int:
    rsf: RandomSurvivalForest = joblib.load(model_pkl)
    order = [
        "n_events_24h",
        "pct_brand_actions",
        "days_since_last_event",
        "opens_last30d",
        "is_bot",
    ]
    X = np.array([[features.get(k, 0.0) for k in order]], dtype=float)
    sf = rsf.predict_survival_function(X, return_array=True)[0]
    risk7 = 1 - sf[min(7, sf.size - 1)]
    return _risk_to_cooldown(risk7)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Email-unsubscribe guardrail RSF")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--csv", required=True, type=Path)
    tr.add_argument("--model", required=True, type=Path)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--n_events_24h", type=int, default=2)
    pr.add_argument("--pct_brand_action", type=float, default=0.1)
    pr.add_argument("--days_since_last_event", type=int, default=3)
    pr.add_argument("--opens_last30d", type=int, default=1)
    pr.add_argument("--is_bot", type=int, choices=[0, 1], default=0)

    ns = ap.parse_args()
    if ns.cmd == "train":
        train_guardrail(ns.csv, ns.model)
    else:
        cd = predict_cooldown(
            ns.model,
            n_events_24h=ns.n_events_24h,
            pct_brand_actions=ns.pct_brand_action,
            days_since_last_event=ns.days_since_last_event,
            opens_last30d=ns.opens_last30d,
            is_bot=ns.is_bot,
        )
        print(f"Cool-down recommendation: {cd} days")


if __name__ == "__main__":
    _cli()
