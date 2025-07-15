# from oss_validation import name_resolution, parsing
from archive import parsing
# TODO: name_resolution module not found in new structure. Update this import when available.


def test_resolve_cue_basic():
    match_name, score, src = name_resolution.resolve_cue("James")
    assert match_name is None or score >= 80


def test_resolve_all_shape():
    df = parsing.parse_abstracts().head(30)
    resolved = name_resolution.resolve_all(df)
    assert {"match_name", "match_score", "match_source"}.issubset(resolved.columns) 