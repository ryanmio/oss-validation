from oss_validation import zone_construction, name_resolution, parsing


def test_zone_build_basic():
    df_cues = parsing.parse_abstracts().head(50)
    resolved = name_resolution.resolve_all(df_cues)
    zones = zone_construction.build_zones(resolved)
    # Should have same unique grant IDs as input subset
    assert zones.grant_id.nunique() == resolved.grant_id.nunique() 