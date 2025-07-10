from __future__ import annotations

import pandas as pd

from oss_validation.anchor_extraction import extract_anchors


def test_extract_anchors_basic():
    data = {
        "cp_raw": [
            # Abbreviation + curly apostrophe, should normalise and match
            "WILLIAM CHAMBERLAYNE … at mouth of Cary's Cr.; adj. Mr. Cocke; 11 Apr. 1732.",
            # Fork with newline split & abbreviation
            "ROBERT BOLLING … at the\nfork of the Nottoway Riv.; 26 July 1722.",
            # No anchor
            "Unnamed abstract with no anchor phrase.",
            # Joins pattern
            "X Branch joins Y Creek near the old mill.",
        ]
    }

    df = pd.DataFrame(data)
    result = extract_anchors(df)

    # Row 0 should detect mouth anchor and expand abbreviation
    row0 = result.iloc[0]
    assert row0.anchor_type == "mouth"
    assert "cary" in row0.anchor_feature_name.lower()

    # Row 1 should detect fork anchor
    row1 = result.iloc[1]
    assert row1.anchor_type == "fork"
    assert "nottoway" in row1.anchor_feature_name.lower()

    # Row 2 should have no anchor
    row2 = result.iloc[2]
    assert pd.isna(row2.anchor_type)
    assert pd.isna(row2.anchor_phrase_raw)

    # Row 3 should detect confluence anchor
    row3 = result.iloc[3]
    assert row3.anchor_type == "confluence"

    # Generic phrases should be dropped
    generic_df = pd.DataFrame({"cp_raw": ["head of a Branch near the line."]})
    gen_res = extract_anchors(generic_df)
    assert pd.isna(gen_res.iloc[0].anchor_type)

    # Ensure no additional anchors beyond expected
    assert len(result) == 4 