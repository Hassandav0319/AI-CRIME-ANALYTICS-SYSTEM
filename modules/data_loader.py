"""
modules/data_loader.py
Handles Kaggle dataset download, preprocessing, and feature engineering
for the Chicago Crime dataset.
"""
import os
import io
import json
import zipfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

KAGGLE_DATASET = "chicago/chicago-crime"
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
SAMPLE_PATH = DATA_DIR / "crime_sample.csv"


# ────────────────────────────────────────────────────────────────────────────
# KAGGLE DOWNLOAD
# ────────────────────────────────────────────────────────────────────────────
def setup_kaggle(api_key: str, username: str) -> bool:
    """Write kaggle.json and validate credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds = {"username": username, "key": api_key}
    cred_path = kaggle_dir / "kaggle.json"
    with open(cred_path, "w") as f:
        json.dump(creds, f)
    os.chmod(cred_path, 0o600)
    return True


def download_kaggle_dataset(dataset: str = KAGGLE_DATASET, dest: Path = DATA_DIR) -> Path | None:
    """Download dataset via Kaggle API. Returns path to CSV or None."""
    try:
        import kaggle  # noqa: F401 – triggers auth check
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
        csvs = list(dest.glob("*.csv"))
        return csvs[0] if csvs else None
    except Exception as e:
        return None


# ────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SAMPLE (when no Kaggle key)
# ────────────────────────────────────────────────────────────────────────────
CRIME_TYPES   = ["ASSAULT", "THEFT", "BATTERY", "NARCOTICS", "BURGLARY",
                 "ROBBERY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE",
                 "WEAPONS VIOLATION", "HOMICIDE"]
LOCATION_DESCS = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK",
                  "PARKING LOT", "ALLEY", "GAS STATION", "RETAIL STORE",
                  "SCHOOL", "RESTAURANT", "PARK"]
DISTRICTS      = list(range(1, 26))

# Chicago city rough bounds
LAT_MIN, LAT_MAX = 41.64, 42.02
LON_MIN, LON_MAX = -87.94, -87.52

HOTSPOT_CENTERS = [
    (41.88, -87.63),  # Chicago Loop
    (41.85, -87.65),  # Near South
    (41.92, -87.68),  # West Side
    (41.77, -87.62),  # Englewood
    (41.79, -87.74),  # Austin
]


def _clustered_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate lat/lon biased toward hotspot centres."""
    lats, lons = [], []
    per_cluster = n // len(HOTSPOT_CENTERS)
    for clat, clon in HOTSPOT_CENTERS:
        lats.extend(np.random.normal(clat, 0.025, per_cluster).clip(LAT_MIN, LAT_MAX))
        lons.extend(np.random.normal(clon, 0.025, per_cluster).clip(LON_MIN, LON_MAX))
    remainder = n - len(lats)
    if remainder > 0:
        lats.extend(np.random.uniform(LAT_MIN, LAT_MAX, remainder))
        lons.extend(np.random.uniform(LON_MIN, LON_MAX, remainder))
    return np.array(lats[:n]), np.array(lons[:n])


def generate_sample_data(n: int = 15_000, save: bool = True) -> pd.DataFrame:
    """
    Generate realistic synthetic Chicago-style crime dataset.
    Mirrors the Kaggle Chicago Crime CSV schema.
    """
    np.random.seed(42)
    end   = datetime.now()
    start = end - timedelta(days=730)  # 2 years of data
    timestamps = pd.to_datetime(
        np.random.uniform(start.timestamp(), end.timestamp(), n), unit="s"
    )

    # Temporal patterns – more crime at night / weekends
    hours   = timestamps.hour.values
    dow     = timestamps.dayofweek.values
    night   = (hours >= 20) | (hours <= 4)
    weekend = dow >= 5

    # Crime type weights (night-skewed for violent crimes)
    base_weights = np.array([0.14, 0.20, 0.12, 0.13, 0.08,
                              0.07, 0.08, 0.09, 0.05, 0.04])
    crime_idx = np.random.choice(len(CRIME_TYPES), n, p=base_weights / base_weights.sum())
    crime_type = [CRIME_TYPES[i] for i in crime_idx]

    lats, lons = _clustered_coords(n)

    # Severity score (used by Random Forest as label)
    violent = {"ASSAULT", "HOMICIDE", "ROBBERY", "BATTERY", "WEAPONS VIOLATION"}
    sev = np.array([
        3 if ct in violent and night[i] else
        2 if ct in violent else
        1 if night[i] or weekend[i] else 0
        for i, ct in enumerate(crime_type)
    ])

    df = pd.DataFrame({
        "ID":             np.arange(1_000_000, 1_000_000 + n),
        "Date":           timestamps,
        "Primary Type":   crime_type,
        "Location Description": np.random.choice(LOCATION_DESCS, n),
        "Arrest":         np.random.choice([True, False], n, p=[0.34, 0.66]),
        "Domestic":       np.random.choice([True, False], n, p=[0.18, 0.82]),
        "District":       np.random.choice(DISTRICTS, n),
        "Ward":           np.random.randint(1, 51, n),
        "Latitude":       lats,
        "Longitude":      lons,
        "Year":           timestamps.year,
        "Severity":       sev,           # 0=low, 1=medium, 2=high, 3=critical
    })
    if save:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(SAMPLE_PATH, index=False)
    return df


# ────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, spatial, and categorical features for model training."""
    df = df.copy()
    if df.empty:
        return df

    if "Date" not in df.columns:
        df["Date"] = datetime.now()

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").fillna(datetime.now())

    df["Hour"]          = df["Date"].dt.hour.fillna(0).astype(int)
    df["DayOfWeek"]     = df["Date"].dt.dayofweek.fillna(0).astype(int)
    df["Month"]         = df["Date"].dt.month.fillna(1).astype(int)
    df["IsWeekend"]     = (df["DayOfWeek"] >= 5).astype(int)
    df["IsNight"]       = ((df["Hour"] >= 20) | (df["Hour"] <= 4)).astype(int)
    df["IsPeakEvening"] = ((df["Hour"] >= 17) & (df["Hour"] <= 22)).astype(int)

    # Lat/Lon buckets → grid cell (200×200 m approx)
    # Ensure Lat/Lon are numeric
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce").fillna(41.8781)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").fillna(-87.6298)

    df["LatBin"] = pd.cut(df["Latitude"],  bins=40, labels=False).fillna(0).astype(int)
    df["LonBin"] = pd.cut(df["Longitude"], bins=40, labels=False).fillna(0).astype(int)

    type_enc = {t: i for i, t in enumerate(CRIME_TYPES)}
    df["CrimeTypeCode"] = df["Primary Type"].map(type_enc).fillna(-1).astype(int)

    loc_enc = {l: i for i, l in enumerate(LOCATION_DESCS)}
    df["LocationCode"] = df["Location Description"].map(loc_enc).fillna(-1).astype(int)

    # Incident density per district
    if "District" not in df.columns:
        df["District"] = 1
    df["District"] = pd.to_numeric(df["District"], errors="coerce").fillna(1).astype(int)

    district_counts = df.groupby("District")["ID"].transform("count")
    max_count = district_counts.max()
    df["DistrictIncidentDensity"] = (district_counts / max_count) if max_count > 0 else 0

    df["ArrestInt"]  = df["Arrest"].fillna(False).astype(int)
    df["DomesticInt"]= df["Domestic"].fillna(False).astype(int)
    return df


FEATURE_COLS = [
    "Hour", "DayOfWeek", "Month", "IsWeekend", "IsNight", "IsPeakEvening",
    "LatBin", "LonBin", "CrimeTypeCode", "LocationCode",
    "DistrictIncidentDensity", "ArrestInt", "DomesticInt", "District",
]
TARGET_COL = "Severity"


def load_or_generate(csv_path: Path | None = None, n_sample: int = 15_000) -> pd.DataFrame:
    """
    Load real Kaggle CSV if provided, else generate synthetic data.
    Runs feature engineering on the result.
    """
    df = None
    if csv_path and Path(csv_path).exists():
        # Try multiple encodings
        for enc in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                df = pd.read_csv(csv_path, low_memory=False, nrows=100_000, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue

    if df is not None:
        # Normalise column names (Kaggle CSV may vary)
        df.columns = [c.strip().title().replace("_", " ") for c in df.columns]

        # Map common aliases (Normalization)
        column_map = {
            "Primary Type": ["Crime Type", "Primary_Type", "Category", "Offense", "Offense Type"],
            "Date": ["Timestamp", "Time", "Occurrence Date", "Created At", "Case Date"],
            "Location Description": ["Location_Description", "Place", "Location Type", "Premise"],
            "District": ["District_Id", "Sector", "Precinct"],
        }
        
        for canonical, aliases in column_map.items():
            if canonical not in df.columns:
                for alias in aliases:
                    # Match case-insensitively and with varying spaces/underscores
                    normalized_aliases = [a.strip().title().replace("_", " ") for a in [alias]]
                    for alt in df.columns:
                        if alt.strip().title().replace("_", " ") == normalized_aliases[0]:
                            df.rename(columns={alt: canonical}, inplace=True)
                            break
                    if canonical in df.columns: break

        # Ensure 'Date' exists after mapping
        if "Date" not in df.columns:
            # Fallback: create a dummy date range
            df["Date"] = pd.date_range(end=datetime.now(), periods=len(df), freq="10min")

        if "Primary Type" not in df.columns:
             df["Primary Type"] = "OTHER"

        if "Severity" not in df.columns:
            violent = {"ASSAULT", "HOMICIDE", "ROBBERY", "BATTERY", "WEAPONS VIOLATION", "WEAPONS_VIOLATION"}
            df["Severity"] = df["Primary Type"].apply(
                lambda x: 2 if str(x).upper() in violent else 1
            )

        df["ID"] = df.get("ID", pd.Series(range(len(df))))

        if "Arrest" not in df.columns:
            df["Arrest"] = False
        if "Domestic" not in df.columns:
            df["Domestic"] = False
        if "Location Description" not in df.columns:
            df["Location Description"] = "STREET"

        for col in ["District", "Ward", "Latitude", "Longitude"]:
            if col not in df.columns:
                if col in ["Latitude", "Longitude"]:
                    df[col] = 41.8781 if col=="Latitude" else -87.6298 # Chicago center
                else:
                    df[col] = 1

    elif SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)
    else:
        df = generate_sample_data(n=n_sample)

    return engineer_features(df)
