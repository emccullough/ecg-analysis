import os
import pandas as pd
from datetime import datetime

ANNOTATION_COLUMNS = [
    "session",
    "r_index",
    "algo_label",
    "human_label",
    "abs_r_time",
    "r_amplitude",
    "rr_interval",
    "qrs_duration",
    "annotated_at",
]


def load_annotations(path="annotations.csv"):
    """Load annotations CSV. Returns empty DataFrame with correct columns if file doesn't exist."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in ANNOTATION_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def save_annotation(session, r_index, algo_label, human_label, metadata, path="annotations.csv"):
    """Append or update a single annotation row. Uses (session, r_index) as key."""
    row = {
        "session": session,
        "r_index": int(r_index),
        "algo_label": algo_label,
        "human_label": human_label,
        "abs_r_time": metadata.get("abs_r_time", ""),
        "r_amplitude": metadata.get("r_amplitude", ""),
        "rr_interval": metadata.get("rr_interval", ""),
        "qrs_duration": metadata.get("qrs_duration", ""),
        "annotated_at": datetime.now().isoformat(),
    }

    df = load_annotations(path)

    mask = (df["session"] == session) & (df["r_index"] == int(r_index))
    if mask.any():
        for col, val in row.items():
            df.loc[mask, col] = val
    else:
        new_row = pd.DataFrame([row])
        if df.empty:
            df = new_row
        else:
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(path, index=False)
    return df


def delete_annotation(session, r_index, path="annotations.csv"):
    """Remove the annotation row for (session, r_index) if it exists."""
    df = load_annotations(path)
    mask = (df["session"] == session) & (df["r_index"] == int(r_index))
    if mask.any():
        df = df[~mask]
        df.to_csv(path, index=False)


def get_annotation_status(session, annotations_df):
    """Return dict with annotation progress for a session."""
    session_df = annotations_df[annotations_df["session"] == session]
    label_counts = session_df["human_label"].value_counts().to_dict()
    return {
        "annotated": len(session_df),
        "by_label": label_counts,
    }
