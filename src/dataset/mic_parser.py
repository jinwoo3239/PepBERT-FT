
import re
import math
import pandas as pd

NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def parse_mic(value):
    """Parse original MIC concentration."""

    if pd.isna(value):
        return None, "missing"

    text = str(value).strip()

    # Normalize decimal commas
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    # Normalize spaces
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    text = text.strip()

    # Censored MIC
    if re.search(r"[<>≤≥]", text):
        return None, "censored"

    # Exact MIC
    if re.fullmatch(NUMBER, text):
        return float(text), "exact"

    # Range MIC
    match = re.fullmatch(
        rf"({NUMBER})\s*[-–~]\s*({NUMBER})",
        text
    )

    if match:
        low, high = map(float, match.groups())

        if low > 0 and high >= low:
            return math.sqrt(low * high), "range_imputed"

        return None, "invalid"

    # Mean ± SD
    match = re.fullmatch(
        rf"({NUMBER})\s*(?:±|\+/-)\s*({NUMBER})",
        text
    )

    if match:
        return float(match.group(1)), "mean"

    return None, "unknown"


def parse_mic_dataframe(df):
    """Parse MIC values in DataFrame."""

    result = df.copy()

    parsed = result["concentration"].apply(parse_mic)

    result["mic_parsed"] = parsed.map(lambda x: x[0])
    result["mic_type"] = parsed.map(lambda x: x[1])

    return result
