from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr

from src.bot_filter import is_bot

RAW = Path("data/raw_events.csv")
if not RAW.exists():
    raise FileNotFoundError(
        "data/raw_events.csv not found.  "
        "Run `make demo-taxonomy` or place the file before launching the API."
    )

events = pd.read_csv(RAW, parse_dates=["event_ts"])
if {"profile_id", "event_name", "purchased_7d"}.difference(events.columns):
    raise ValueError("raw_events.csv missing required columns")

events["is_sms"] = events["event_name"].eq("SMS Sent").astype(int)
events["label"] = events["purchased_7d"]

pivot = (
    events.pivot_table(
        index=["profile_id"],
        columns="is_sms",
        values="label",
        aggfunc="mean",
        fill_value=0,
    )
    .rename(columns={0: "p_no_sms", 1: "p_sms"})
    .assign(uplift=lambda d: d["p_sms"] - d["p_no_sms"])
    .reset_index()
)

TOP_PROFILES = pivot.loc[pivot["uplift"] > 0.05, "profile_id"].tolist()

app = FastAPI(
    title="touchpoint-lab demo",
    description="SMS uplift & bot checker (POC)",
    version="0.1.0",
)


class BotCheckIn(BaseModel):
    email: EmailStr
    opens: int = 0
    clicks: int = 0


class BotCheckOut(BaseModel):
    is_bot: bool
    reason: str


@app.get(
    "/sms/recommendations",
    response_model=List[int],
    summary="Profiles with uplift > 5 pp",
)
def sms_recommendations(limit: int = Query(100, ge=1, le=1_000)):
    if not TOP_PROFILES:
        raise HTTPException(
            status_code=503,
            detail="Uplift table is empty – no SMS events in dataset.",
        )
    return TOP_PROFILES[:limit]


@app.post(
    "/bot/check",
    response_model=BotCheckOut,
    summary="Heuristic bot detector",
)
def bot_check(body: BotCheckIn):
    try:
        flag = is_bot(body.opens, body.clicks, body.email)
    except Exception as exc:  # guard against unexpected types
        raise HTTPException(status_code=400, detail=str(exc))

    reason = (
        "temp-mail domain or extreme engagement" if flag else "pattern looks human"
    )
    return BotCheckOut(is_bot=flag, reason=reason)
