# merge_csv_to_parquet.py
# Step 0 of the pipeline: dataset loading, CSV merging and label assignment.
#
# Input : the original CIC IoT-DIAD 2024 flow-based CSV files placed under data/raw/
#         (any sub-folder structure, see README).
# Output: data/attacks.parquet and data/Benign.parquet
#         Both contain the original columns plus a "Label" column whose value is
#         exactly "Attack" or "Benign". This is what prepare_iot_diad_train_test_stream.py expects.
#
# Label rule (in this order):
#   1. If the CSV has a "Label" column: "benign" (case-insensitive) -> Benign, anything else -> Attack.
#   2. Otherwise: if the word "benign" appears in the file path -> Benign, else Attack.
#
# Row order: files are processed in natural sorted path order and rows are kept in file
# order. No re-sorting is done. The later chronological 70/30 split depends on this order.
#
# The CSVs are read in chunks, so memory use stays small even for ~27 million rows.

import os
import re
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import RAW_DIR, ATTACK_PATH, BENIGN_PATH

CHUNK_ROWS = 500_000
TEXT_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]


def natural_key(path: str):
    # natural sort so that part_2.csv comes before part_10.csv
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def list_csv_files(folder):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(".csv"):
                files.append(os.path.join(root, n))
    files.sort(key=natural_key)
    return files


def to_arrow_table(df: pd.DataFrame, columns, feature_cols):
    df = df.reindex(columns=columns)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return pa.Table.from_pandas(df, preserve_index=False)


def main():
    files = list_csv_files(RAW_DIR)
    if not files:
        print(f"[ERROR] No CSV files found under: {RAW_DIR}")
        print("        Download the dataset and put the flow CSV files there (see README).")
        sys.exit(1)

    os.makedirs(os.path.dirname(ATTACK_PATH), exist_ok=True)
    writers = {}
    counts = {"Attack": 0, "Benign": 0}
    columns = None
    feature_cols = None
    schema = None

    for fi, path in enumerate(files, 1):
        print(f"[{fi}/{len(files)}] {os.path.relpath(path, RAW_DIR)}")
        path_says_benign = "benign" in path.lower()

        for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
            chunk.columns = [str(c).strip() for c in chunk.columns]

            if columns is None:
                columns = list(chunk.columns)
                if "Label" not in columns:
                    columns.append("Label")
                feature_cols = [c for c in columns if c not in TEXT_COLS]
                print(f"       columns: {len(columns)} (numeric: {len(feature_cols)})")

            if "Label" in chunk.columns:
                is_benign = chunk["Label"].astype(str).str.strip().str.lower() == "benign"
            else:
                is_benign = pd.Series(path_says_benign, index=chunk.index)
            chunk["Label"] = np.where(is_benign, "Benign", "Attack")

            missing = [c for c in columns if c not in chunk.columns]
            if missing:
                print(f"       [WARN] file lacks columns {missing[:5]}, filled with NaN")

            for name, mask in (("Attack", ~is_benign), ("Benign", is_benign)):
                part = chunk[mask.values]
                if part.empty:
                    continue
                table = to_arrow_table(part, columns, feature_cols)
                if schema is None:
                    schema = table.schema
                table = table.cast(schema)
                if name not in writers:
                    out = ATTACK_PATH if name == "Attack" else BENIGN_PATH
                    writers[name] = pq.ParquetWriter(out, schema, compression="snappy")
                writers[name].write_table(table)
                counts[name] += len(part)

    for w in writers.values():
        w.close()

    print("\nDone.")
    print(f"  Attack rows: {counts['Attack']:,}  -> {ATTACK_PATH}")
    print(f"  Benign rows: {counts['Benign']:,}  -> {BENIGN_PATH}")
    if counts["Attack"] == 0 or counts["Benign"] == 0:
        print("[WARN] One class is empty. Check the label rule at the top of this file.")


if __name__ == "__main__":
    main()
