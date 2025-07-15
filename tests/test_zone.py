# from oss_validation import zone_construction, name_resolution, parsing
from archive import parsing
# TODO: zone_construction and name_resolution modules not found in new structure. Update these imports when available.


def test_zone_build_basic():
    df_cues = parsing.parse_abstracts().head(50)
    resolved = name_resolution.resolve_all(df_cues)
    zones = zone_construction.build_zones(resolved)
    # Should have same unique grant IDs as input subset
    assert zones.grant_id.nunique() == resolved.grant_id.nunique() 