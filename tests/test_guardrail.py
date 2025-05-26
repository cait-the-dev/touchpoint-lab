from pathlib import Path
import pytest
from src.guardrail_model import train_guardrail, predict_cooldown

CSV = Path("data/raw_events.csv")


@pytest.mark.skipif(not CSV.exists(), reason="sample events missing")
def test_guardrail_pipeline(tmp_path):
    model_path = tmp_path / "guardrail_test.pkl"

    train_guardrail(CSV, model_path)
    assert model_path.exists(), "Model file was not created"

    cd = predict_cooldown(model_path, n_events_24h=4, pct_brand_actions=0.05)
    assert isinstance(cd, int)
    assert 0 <= cd <= 30