"""Extract grantee name, acreage, year and county from Cavaliers & Pioneers Volume 3 abstracts.

Output CSV: cp_grants.csv with columns:
    grant_id, name_std, acreage, year, county_text, raw_entry

We rely on regex rules that capture the first line of the abstract which has the
format:
    <GRANTEE>, <ACREAGE> acs., <County> Co; ... <DATE YEAR>

The extract is best-effort; any record missing a cue is left blank so that we
can later filter / manually inspect.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

from archive import parsing

from oss_preprocessing import config

OUT_CSV = config.PROCESSED_DIR / "cp_grants.csv"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Title words to strip
_TITLE_RE = re.compile(r"^(?:COL\.|CAPT\.|MR\.?,|MRS\.|LT\.|MAJ\.|GEN\.|DR\.)\s+", re.IGNORECASE)
# Name ends at first comma
_NAME_RE = re.compile(r"^([A-Za-z .'&\-]{3,}?),")
# Acreage like "5000a", "400 acres", "99.5 acs." etc.
_ACRE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*a(?:c|cs|cres|res)?\.?", re.IGNORECASE)
# 4-digit year pattern
_YEAR_RE = re.compile(r"(16|17|18)\d{2}")
# Capture words before Co/County/City/Go (handle OCR corruption)
_COUNTY_RE = re.compile(
    r"([A-Z][A-Za-z &'\.\-:8é!]{2,15})"        # candidate county words
    r"\s*[,:;']?\s*"
    r"(?:Co|Go)\b"                              # explicit word-boundary after Co/Go
    r"(?:\s+[a-z]{1,3})?"                       # allow trailing OCR garbage like 'cn'
    r"\s*[\.,:;']?",
    re.IGNORECASE,
)

# Canonicalise common abbreviations / shorthand seen in OCR
_COUNTY_ABBR_MAP = {
        # King & Queen
    "K&Q": "King & Queen",
    "KANDQ": "King & Queen", 
    "KQ": "King & Queen",
    "K & Q": "King & Queen",
    "K 8& Q": "King & Queen",  # OCR fix
    # Henrico
    "HNCO": "Henrico",
    "HN": "Henrico",
    # Spotsylvania
    "SPCO": "Spotsylvania",
    "SP": "Spotsylvania",
    "SPOTSYL": "Spotsylvania",
    "SPOTSYLVA": "Spotsylvania",
    "SPOTSYLV": "Spotsylvania",
    "SPOTSYLVANIA": "Spotsylvania",
    # Prince George variants
    "PR GEO": "Prince George",
    "PR. GEO": "Prince George",
    "PR GEO CO": "Prince George",
    "PR GEO.": "Prince George",
    "PR. GEO.": "Prince George",
    "PRGEO": "Prince George",
    "GEO": "Prince George",
    # Prince William shortcuts
    "PR WM": "Prince William",
    "PR. WM": "Prince William",
    # Brunswick
    "BRUNSW": "Brunswick",
    "BRUNSWCO": "Brunswick",
    # Appomattox
    "APCO": "Appomattox",
    # Frederick abbreviation
    "FREDERIC": "Frederick",
    "FREDER": "Frederick",
    # Charles City abbreviations
    "CHAS": "Charles City",
    "CHASCITY": "Charles City", 
    "CHARLESCITY": "Charles City",
    "CHAS CITY": "Charles City",
    # Princess Anne
    "PRANNE": "Princess Anne",
    "PRINCEANNE": "Princess Anne",
    "PRINCESSAN": "Princess Anne",
    "ANNE": "Princess Anne",
    # Isle of Wight
    "WIGHT": "Isle Of Wight",
    "ISLEWIGHT": "Isle Of Wight",
    "ISOFWIGHT": "Isle Of Wight",
    "ISLEOFWIGHT": "Isle Of Wight",
    "IS OF WIGHT": "Isle Of Wight",
    "ISOFWIGHT": "Isle Of Wight",
    # James City
    "JAMES": "James City",
    "JAMESCITY": "James City",
    "JAS": "James City",
    # Gloucester misspell
    "GLOCESTER": "Gloucester",
    "GLOUCESTER": "Gloucester",
    # Nansemond misspell
    "NANSAMOND": "Nansemond",
    "NANSEMOND": "Nansemond",
    "NANSEMND": "Nansemond",  # OCR fix
    "NANSEMND": "Nansemond",  # OCR é -> e conversion
        # Norfolk abbreviations
    "NORF": "Norfolk",
    "NORFOL": "Norfolk",
    "NORFOLK": "Norfolk",
    # Northampton abbreviations  
    "NAMPTON": "Northampton",
    "NORTHAMPTON": "Northampton",
    "AMPTON": "Northampton",  # Fix truncation issue
    # King William
    "KINGWM": "King William",
    "KINGWILLIAM": "King William",
    # Surry variants
    "SURRV": "Surry",
    "SURRY": "Surry",
    # Essex variants
    "ESSEX": "Essex",
    "ESX": "Essex",
    "ESSX": "Essex",  # OCR fix
    # Elizabeth City
    "ELIZ": "Elizabeth City",
    "ELIZABETH": "Elizabeth City",
    "ELIZ CITY": "Elizabeth City",
    "ELIZCITY": "Elizabeth City",
    "ELIZA CITY": "Elizabeth City",
    "ELIZA": "Elizabeth City",
    # King William truncation fixes
    "WM": "King William", 
    "KING WM": "King William",
    # James City truncation fixes  
    "JAS CITY": "James City",
    "JASCITY": "James City",
    # Handle common "Of X" patterns
    "OF WIGHT": "Isle Of Wight",
    "OF NANSEMOND": "Nansemond",
    "OF SURRY": "Surry", 
    "OF HANOVER": "Hanover",
    "OF PR GEO": "Prince George",
    "OF K & Q": "King & Queen",
    "OF HENRICO": "Henrico",
    "OF MIDDLESEX": "Middlesex",
    "OF JAMES CITY": "James City",
    "OF YORK": "York",
    "OF GLOUCESTER": "Gloucester",
    "OF NANSAMOND": "Nansemond",
    "OF NENSEMOND": "Nansemond",
    # Handle "In X" patterns  
    "IN K & Q": "King & Queen",
    "IN HENRICO": "Henrico",
    "IN KING WM": "King William",
    "IN HANOVER": "Hanover",
    "IN SPOTSYL": "Spotsylvania",
    "IN PR GEO": "Prince George",
    "IN ESSEX": "Essex",
    "IN CHAS CITY": "Charles City",
    "IN ACCOMACK": "Accomack",
    "IN SURRY": "Surry",
    "IN ELIZA": "Elizabeth City",
    "IN GLOUCESTER": "Gloucester",
    "IN NEW KENT": "New Kent",
    "IN KING & QUEEN": "King & Queen",
    "IN WARWICK": "Warwick",
    "IN GOOCHLAND": "Goochland",
    # Handle "Acs X" patterns (acres in X county)
    "ACS K & Q": "King & Queen",
    "ACS NORF": "Norfolk",
    "ACS MIDDLESEX": "Middlesex", 
    "ACS HENRICO": "Henrico",
    "ACS YORK": "York",
    "ACS CHAS CITY": "Charles City",
    "ACS NEW KENT": "New Kent",
    "ACS NANSEMOND": "Nansemond",
    "ACS ACCOMACK": "Accomack",
    "ACS SURRY": "Surry",
    "ACS WARWICK": "Warwick",
    "ACS GOOCHLAND": "Goochland",
    "ACS GLOCESTER": "Gloucester",
    "ACS JAS CITY": "James City",
    "ACS PR ANNE": "Princess Anne",
    # Common misspellings
    "HENTICO": "Henrico",
    "HENSICO": "Henrico",
    "SURTY": "Surry",
    "GLOSTER": "Gloucester",
    "GLOUSTER": "Gloucester",
    "PR ANN": "Princess Anne",
    "SPOTSYLY": "Spotsylvania",
    "SPOTYL": "Spotsylvania",
    "SPORSYL": "Spotsylvania",
    "SPOTSVL": "Spotsylvania",
    "SPOTSVLV": "Spotsylvania",
    "SPOTSVLY": "Spotsylvania",
    "IA SPOTSYL": "Spotsylvania",
    "YORKE": "York",
    "HENRI": "Henrico",
    "HENRICE": "Henrico",
    "NEW KENT": "New Kent",
    "NEWKENT": "New Kent",
    "NEW-KENT": "New Kent",
    "BRANSWICK": "Brunswick",
    "PZ ANN": "Princess Anne",
    "AC COMACK": "Accomack",
    "INKQ": "King & Queen",
    "IN K Q": "King & Queen",
    "IN K   Q": "King & Queen",
    "ACS K Q": "King & Queen",
    # Additional mappings requested
    "SPOTSY": "Spotsylvania",  # Handle "Spotsy!. Co."
    "IGHT": "Isle Of Wight",   # Handle truncated "Wight"
    "OF WICHT": "Isle Of Wight",  # OCR corruption of "Of Wight"
    "OF K & Q": "King & Queen",
    "IN PR ANNE": "Princess Anne",
}

# ---------------------------------------------------------------------------

# ---- Canonical county list (modern spellings) ---------------------------
_CANONICAL_COUNTIES = {
    'HENRICO','PRINCE GEORGE','SURRY','ISLE OF WIGHT','SPOTSYLVANIA','HANOVER','BRUNSWICK',
    'NANSEMOND','KING WILLIAM','GOOCHLAND','KING & QUEEN','NEW KENT','ESSEX','NORFOLK',
    'PRINCESS ANNE','CHARLES CITY','MIDDLESEX','JAMES CITY','GLOUCESTER','ACCOMACK',
    'CAROLINE','WARWICK','YORK','NORTHAMPTON','ELIZABETH CITY'
}

def _normalise_name(raw: str) -> str:
    raw = _TITLE_RE.sub("", raw).strip()
    if "," in raw:
        raw = raw.split(",", 1)[0]
    return " ".join(w.capitalize() for w in raw.split())


def _normalise_county(raw: str) -> str:
    # Remove punctuation (keep spaces), collapse multiple spaces, uppercase
    key = re.sub(r"[^A-Za-z ]", "", raw)
    key = re.sub(r"\s+", " ", key).strip().upper()
    
    # Try exact match in abbreviation map first
    mapped = _COUNTY_ABBR_MAP.get(key)
    if mapped:
        return mapped
    
    # Try without spaces for compound abbreviations
    key_no_space = key.replace(" ", "")
    mapped = _COUNTY_ABBR_MAP.get(key_no_space)
    if mapped:
        return mapped
    
    # Title case the original raw input as fallback (preserving original spelling)
    candidate = " ".join(w.capitalize() for w in raw.split())

    # Final validation: if the candidate is not one of our canonical counties, discard
    if candidate.upper() not in _CANONICAL_COUNTIES:
        return None

    return candidate


def _extract_from_text(text: str) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[str]]:
    """Return (name, acreage, year, county) from an abstract string."""
    name: Optional[str] = None
    acreage: Optional[float] = None
    year: Optional[int] = None
    county: Optional[str] = None

    # Clean text first line region
    clean = text.replace("\u2014", "-").replace("\u2013", "-")  # dashes
    clean = re.sub(r"\s+", " ", clean.strip())

    # Examine a longer slice (800 chars) to catch county / year that appear later
    head = clean[:800]

    m_name = _NAME_RE.match(head)
    if m_name:
        candidate = _normalise_name(m_name.group(1))
        if candidate.lower() == "same":
            candidate = None
        name = candidate

    # Acreage first (needed for year logic)
    m_ac = _ACRE_RE.search(head)
    acre_end = 0
    if m_ac:
        try:
            acreage = float(m_ac.group(1))
        except ValueError:
            pass
        acre_end = m_ac.end()

    # ---- Year extraction -------------------------------------------------
    year_match = None
    if m_ac:
        year_match = _YEAR_RE.search(head[m_ac.end():])
    if year_match is None:
        year_match = _YEAR_RE.search(head)
    if year_match is None and m_ac:
        year_match = _YEAR_RE.search(clean[m_ac.end():])
    if year_match is None:
        # OCR fix patterns e.g. i725 or (727
        ocr_match = re.search(r"[iI\(](6|7|8)\d{2}", head)
        if ocr_match:
            yr_str = '1' + ocr_match.group(1) + ocr_match.group(0)[2:]
            try:
                yr_int = int(yr_str)
                if 1600 <= yr_int <= 1932:
                    year = yr_int
            except ValueError:
                pass
    else:
        yr_int = int(year_match.group(0))
        if 1600 <= yr_int <= 1932:
            year = yr_int

    # ---- Fallback name ---------------------------------------------------
    if name is None:
        # take text before acreage token
        if m_ac:
            pre = head[:m_ac.start()].strip()
            pre = re.sub(r"\bof\s+[A-Z].*", "", pre, flags=re.IGNORECASE)
            pre = re.split(r",|\.", pre)[0]
            pre = pre.strip()
            if 3 <= len(pre) <= 60:
                candidate = _normalise_name(pre)
                if candidate.lower() != "same":
                    name = candidate

    def _clean_raw_cty(raw: str) -> str:
        raw = raw.replace(".", "")
        # OCR corrections
        raw = raw.replace("8&", "&").replace("::", "s")  # Fix "K. 8& Q" and "Es::x"
        raw = raw.replace("é", "e").replace("ém", "e")  # Fix "Nansémnd"
        raw = re.sub(r"['`]", "", raw)  # Remove stray apostrophes/backticks
        raw = re.sub(r"\s+", " ", raw)  # Normalize spaces
        
        # If phrase contains ' of ', keep part after last 'of'
        if ' of ' in raw.lower():
            raw = raw.split(' of ')[-1]
            
        raw = raw.strip()
        
        # Remove leading hyphens that come from OCR issues
        raw = raw.lstrip('-')
        
        # Skip tokens that refer to parish
        if raw.upper().startswith(('UP', 'LOW')):
            tokens = raw.split()
            if len(tokens) > 1:
                raw = ' '.join(tokens[1:])
                
        # Handle "Is, of Wight" pattern - prefer full form
        if raw.lower().startswith('is') and 'wight' in raw.lower():
            return 'Is of Wight'
        
        # Handle "Acs X" pattern - remove "Acs" prefix
        if raw.upper().startswith(('ACS ', 'AES ')):
            raw = raw[4:].strip()

        # Remove stray leading percentage or numbers (e.g., "427% acs.,")
        raw = re.sub(r'^[\d% ]+', '', raw)
        
        return raw

    # County extraction: prioritize post-acreage mentions
    county_candidates = []
    
    # First pass: look for counties after acreage mention
    if acre_end > 0:
        for m in _COUNTY_RE.finditer(head):
            if m.start() > acre_end:
                raw_cty = _clean_raw_cty(m.group(1))
                # Skip "same" as it's not a real county name
                if raw_cty.lower() == "same":
                    continue
                norm_cty = _normalise_county(raw_cty)
                if norm_cty and norm_cty.upper() not in {"PAR", "PARISH", "SAME"}:
                    county_candidates.append(norm_cty)
    
    # Second pass: if no post-acreage county found, search anywhere in the text
    if not county_candidates:
        for m in _COUNTY_RE.finditer(head):
            raw_cty = _clean_raw_cty(m.group(1))
            # Skip "same" as it's not a real county name
            if raw_cty.lower() == "same":
                continue
            norm_cty = _normalise_county(raw_cty)
            if norm_cty and norm_cty.upper() not in {"PAR", "PARISH", "SAME"}:
                county_candidates.append(norm_cty)
    
    # Third pass: fallback to full text search
    if not county_candidates:
        for m in _COUNTY_RE.finditer(clean):
            raw_cty = _clean_raw_cty(m.group(1))
            # Skip "same" as it's not a real county name
            if raw_cty.lower() == "same":
                continue
            norm_cty = _normalise_county(raw_cty)
            if norm_cty and norm_cty.upper() not in {"PAR", "PARISH", "SAME"}:
                county_candidates.append(norm_cty)
    
    # Take the first valid candidate
    if county_candidates:
        county = county_candidates[0]

    return name, acreage, year, county


# ---------------------------------------------------------------------------


def build_cp_grants_table(out_csv: Path = OUT_CSV) -> pd.DataFrame:
    df_raw = parsing._read_raw_csv()  # type: ignore

    records: List[Dict[str, object]] = []
    misses = 0
    same_count = 0
    prev_name: Optional[str] = None
    prev_year: Optional[int] = None
    prev_county: Optional[str] = None

    for _, row in df_raw.iterrows():
        name, acreage, year, county = _extract_from_text(row.raw_entry)

        # --- Handle "Same ..." chains ------------------------------------
        low_head = row.raw_entry.lstrip().lower()[:120]
        is_same_entry = low_head.startswith("same")
        if is_same_entry:
            same_count += 1
            same_loc = ("location" in low_head or "loc." in low_head or "co." in low_head or " county" in low_head)
            same_date = "date" in low_head
            
            # For Same entries, always propagate name if we didn't extract one
            if name is None and prev_name:
                name = prev_name
            
            # For county: if "same location" mentioned OR no county extracted, use previous county
            if (same_loc or county is None) and prev_county:
                county = prev_county
            
            # For year: if "same date" mentioned OR no year extracted, use previous year  
            if (same_date or year is None) and prev_year:
                year = prev_year

        if name is None or acreage is None or year is None or county is None:
            misses += 1
        records.append({
            "grant_id": row.grant_id,
            "name_std": name,
            "acreage": acreage,
            "year": year,
            "county_text": county,
            "raw_entry": row.raw_entry,
        })

        # Update rolling previous non-null values (but only for non-"Same" entries to avoid pollution)
        if not is_same_entry:
            if name and name.lower() != 'same':
                prev_name = name
            if year:
                prev_year = year
            if county:
                prev_county = county

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    extracted = len(df) - misses
    print(f"C&P abstracts parsed: {extracted}/{len(df)} fully-extracted rows. → {out_csv.relative_to(config.ROOT_DIR)}")
    print(f"'Same' entries found: {same_count}")
    if misses:
        print(f"WARNING: {misses} rows missing at least one cue (left blank).")
        # Show breakdown of what's missing
        missing_name = sum(1 for r in records if r['name_std'] is None)
        missing_acreage = sum(1 for r in records if r['acreage'] is None)
        missing_year = sum(1 for r in records if r['year'] is None)
        missing_county = sum(1 for r in records if r['county_text'] is None)
        print(f"  Missing: {missing_name} names, {missing_acreage} acreage, {missing_year} years, {missing_county} counties")
    return df


if __name__ == "__main__":  # pragma: no cover
    build_cp_grants_table() 