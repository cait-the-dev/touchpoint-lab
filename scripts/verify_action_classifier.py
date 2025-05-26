from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.action_classifier import load_labels, train, predict, _norm

RAW    = Path("data/raw_events.csv")
LABELS = Path("data/labeled_event_actions_aug.csv")
MODEL  = Path("models/event_action_clf.pkl")

print("\n[1] prepare supervised dataframe …")
lab = load_labels(LABELS)
raw = (
    pd.read_csv(RAW, usecols=["event_name"])
      .assign(event_name=lambda d: _norm(d["event_name"]))
)
df = raw.merge(lab, on="event_name", how="left")
assert df["label"].notna().all(), "Some raw rows lack a label!"

print("[2] 90/10 train vs test row split …")
X_train, X_test, y_train, y_test = train_test_split(
    df["event_name"], df["label"], test_size=0.1, stratify=df["label"], random_state=42
)

clf = train(LABELS, MODEL)
print(classification_report(y_test, clf.predict(X_test), digits=3))
print("Confusion:\n", confusion_matrix(y_test, clf.predict(X_test)))

print("[3] Leave‑one‑name‑out F1 check …")
unique_names = lab["event_name"].unique()

scores = []
for nm in unique_names:
    mask_train = df["event_name"] != nm
    mask_test  = ~mask_train

    if mask_test.sum() == 0:
        print(f"No rows for '{nm}' – skipping")
        continue

    tmp_labels = Path("/tmp/tmp_labels.csv")
    df.loc[mask_train, ["event_name", "label"]].drop_duplicates().to_csv(tmp_labels, index=False)
    lono_clf = train(tmp_labels, Path("/tmp/tmp_model.pkl"))

    y_true = df.loc[mask_test, "label"].values
    y_pred = [predict(lono_clf, txt) for txt in df.loc[mask_test, "event_name"]]

    scores.append(
        f1_score(y_true, y_pred, pos_label="BRAND", zero_division=1)
    )

if scores:
    print(f"Mean LONO F1 over {len(scores)} names: {sum(scores)/len(scores):.3f}")
else:
    print("No names had held-out rows; LONO skipped.")
