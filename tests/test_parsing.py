from oss_validation import parsing


def test_extract_cues_simple():
    text = "Situated on the North side of James River near Nottoway Creek."
    cues = parsing.extract_cues(text)
    assert "James" in cues and "Nottoway" in cues


def test_parse_abstracts_schema():
    df = parsing.parse_abstracts().head(50)
    assert set({"grant_id", "cue", "raw_text"}).issubset(df.columns)
    # At least some cues extracted
    assert len(df) > 0 