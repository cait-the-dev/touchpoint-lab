import itertools

import pandas as pd

LABELS = "data/labeled_event_actions.csv"
OUT = "data/labeled_event_actions_aug.csv"

cust_tokens = [
    ["opened", "clicked", "viewed", "placed", "added", "started", "completed"],
    ["email", "product", "order", "cart", "quiz"],
]

brand_tokens = [
    ["email", "sms", "push", "notification", "order"],
    ["sent", "delivered", "shipped", "refunded"],
]


def synth_rows(tokens, label, n):
    combos = itertools.islice(itertools.product(*tokens), n)
    return pd.DataFrame(
        {"event_name": [" ".join(c).title() for c in combos], "label": label}
    )


orig = pd.read_csv(LABELS)
orig = orig.rename(columns={"cls": "label"}) if "cls" in orig else orig

syn_cust = synth_rows(cust_tokens, "CUSTOMER", 25)
syn_brand = synth_rows(brand_tokens, "BRAND", 25)

aug = (
    pd.concat([orig[["event_name", "label"]], syn_cust, syn_brand])
    .drop_duplicates()
    .reset_index(drop=True)
)

aug.to_csv(OUT, index=False)
print(f"Augmented label file saved → {OUT}  ({len(orig)} → {len(aug)} rows)")
