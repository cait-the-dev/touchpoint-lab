from src.event_taxonomy import (
    load_events,
    canonicalise,
    build_feature_matrix,
)

CSV = "data/raw_events.csv"

def test_pipeline_smoke():
    df = load_events(CSV)
    assert not df.empty
    assert {"event_name", "purchased_7d"}.issubset(df.columns)

    df2 = canonicalise(df.copy())
    assert "canonical_event" in df2
    assert df2["canonical_event"].nunique() > 5
    assert "vec" in df2.columns and df2["vec"].notna().all()

    X, y = build_feature_matrix(df2.head(500))
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] >= 3
    assert "is_brand_action" in X.columns
