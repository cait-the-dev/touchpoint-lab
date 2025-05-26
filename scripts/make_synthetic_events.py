import datetime
import random

import numpy as np
import pandas as pd

n_rows, n_profiles = 20_000, 3_000
base_events = [
    "Opened Email",
    "Email Open",
    "Clicked Email",
    "Placed Order",
    "Viewed Product",
    "Active on Site",
    "Unsubscribed",
    "Clicked email to unsubscribe",
    "SMS Sent",
    "SMS Delivered",
    "Octane: quiz completed: Find the perfect eye shadow",
    "Black Crow Identified",
    "Viewed Product XYZ",
    "blotout viewed product XYZ",
]


def perturb(e):
    if random.random() < 0.2:
        e = e.lower().title()
    if random.random() < 0.2:
        e += random.choice(["  ", "", "   "])
    return e


events = [perturb(e) for e in base_events]

start = datetime.datetime(2025, 1, 1)
rows = []
for _ in range(n_rows):
    rows.append(
        {
            "profile_id": random.randint(1, n_profiles),
            "event_ts": start
            + datetime.timedelta(seconds=random.randint(0, 60 * 60 * 24 * 120)),
            "event_name": random.choice(events),
            "purchased_7d": 1 if "Order" in events else np.random.binomial(1, 0.05),
        }
    )

pd.DataFrame(rows).to_csv("sample_events.csv", index=False)
