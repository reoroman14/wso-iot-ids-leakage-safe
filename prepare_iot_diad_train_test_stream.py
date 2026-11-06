import os
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from config import (ATTACK_PATH, BENIGN_PATH, OUT_DIR, TRAIN_DIR, TEST_DIR,
                    TRAIN_RATIO, DROP_COLS)  # paths and settings come from config.py

def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def normalize_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df["y"] = (df["Label"].astype(str) == "Attack").astype(np.int8)
    df = df.drop(columns=["Label"], errors="ignore")
    return df

def write_part(df: pd.DataFrame, out_dir: str, prefix: str, idx: int):
    table = pa.Table.from_pandas(df, preserve_index=False)
    out_path = os.path.join(out_dir, f"{prefix}_part_{idx:05d}.parquet")
    pq.write_table(table, out_path, compression="snappy")

def stream_split_by_rows(parquet_path: str, prefix: str, train_ratio: float):
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    train_rows_target = int(total_rows * train_ratio)

    train_parts = 0
    test_parts = 0
    seen_rows = 0

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        df = normalize_and_label(df)

        n = len(df)
        # How many rows from this group go to train?
        remaining_train = max(train_rows_target - seen_rows, 0)
        take_train = min(remaining_train, n)

        if take_train > 0:
            df_train = df.iloc[:take_train].copy()
            write_part(df_train, TRAIN_DIR, prefix, train_parts)
            train_parts += 1

        if take_train < n:
            df_test = df.iloc[take_train:].copy()
            write_part(df_test, TEST_DIR, prefix, test_parts)
            test_parts += 1

        seen_rows += n
        del df, table

    print(f"{prefix}: total_rows={total_rows}, train_target={train_rows_target}")
    print(f"{prefix}: wrote {train_parts} train parts and {test_parts} test parts.")

def summarize(dataset_dir: str, name: str):
    dataset = ds.dataset(dataset_dir, format="parquet")
    total = 0
    y0 = 0
    y1 = 0
    scanner = dataset.scanner(columns=["y"])
    for batch in scanner.to_batches():
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        total += len(arr)
        s = int(arr.sum())
        y1 += s
        y0 += (len(arr) - s)
    print(f"\n{name} rows: {total}")
    print(f"{name} y counts: 0={y0}, 1={y1}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reset_dir(TRAIN_DIR)
    reset_dir(TEST_DIR)

    print("Streaming ATTACK file (row-count split)...")
    stream_split_by_rows(ATTACK_PATH, "attack", TRAIN_RATIO)

    print("\nStreaming BENIGN file (row-count split)...")
    stream_split_by_rows(BENIGN_PATH, "benign", TRAIN_RATIO)

    summarize(TRAIN_DIR, "TRAIN")
    summarize(TEST_DIR, "TEST")

    print("\nSaved datasets:")
    print(" Train:", TRAIN_DIR)
    print(" Test :", TEST_DIR)

if __name__ == "__main__":
    main()
