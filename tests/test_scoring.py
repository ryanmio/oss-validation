from oss_validation import scoring, zone_construction, name_resolution, parsing
import pandas as pd


def test_scoring_output_columns():
    df_cues = parsing.parse_abstracts().head(30)
    resolved = name_resolution.resolve_all(df_cues)
    zones = zone_construction.build_zones(resolved)

    # fake centroid list
    cent_df = pd.DataFrame({
        "grant_id": zones.grant_id,
        "lat": [37.0] * len(zones),
        "lon": [-78.0] * len(zones),
    })
    cent_gdf = scoring._point_gdf(cent_df)
    res = scoring.score(zones, cent_gdf)
    expected_cols = {"grant_id", "in_zone", "overshoot_km", "zone_area_km2", "num_cues"}
    assert expected_cols.issubset(res.columns) 