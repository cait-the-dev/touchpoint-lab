from pathlib import Path

import pandas as pd

from src.action_classifier import predict, train

LABELS = Path("data/labeled_event_actions_aug.csv")
MODEL = Path("models/test_action_clf.pkl")


def test_end_to_end(tmp_path):
    model_path = tmp_path / "clf.pkl"
    clf = train(LABELS, model_path)

    assert model_path.exists()
    assert predict(model_path, "SMS Sent") in {"BRAND", "CUSTOMER"}

    df = pd.read_csv(LABELS)
    preds = clf.predict(df["event_name"])
    acc = (preds == df["label"].str.upper()).mean()
    assert acc >= 0.99
