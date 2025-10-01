#!/usr/bin/env python3
"""
Trim a time-series CSV to keep only the most recent fraction of rows.

Usage:
  python trim_dataset.py --input X_BTCUSD_90d_5min_features.csv --output X_BTCUSD_recent_half.csv --fraction 0.5

Options:
  --input      Path to source CSV.
  --output     Path to output CSV (will be overwritten if exists).
  --fraction   Fraction of most recent rows to keep (0 < f <= 1). Default: 0.5
  --date-col   Optional explicit date column name (otherwise auto-detect 'date' or 'timestamp').
"""
import argparse
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the most recent fraction of a time-series CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction of most recent rows to keep (0<f<=1)")
    parser.add_argument("--date-col", default=None, help="Optional date column name (auto-detect if omitted)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not (0 < args.fraction <= 1):
        print(f"❌ fraction must be in (0,1], got {args.fraction}", file=sys.stderr)
        return 2

    # Load CSV
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"❌ Failed to read input CSV: {e}", file=sys.stderr)
        return 1

    # Detect date column
    date_col = args.date_col
    if date_col is None:
        if "date" in df.columns:
            date_col = "date"
        elif "timestamp" in df.columns:
            date_col = "timestamp"
        else:
            print("❌ Could not detect a date column. Provide --date-col (e.g., 'timestamp').", file=sys.stderr)
            return 3

    # Parse and sort by date ascending
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception as e:
        print(f"❌ Failed parsing date column '{date_col}': {e}", file=sys.stderr)
        return 4

    before = len(df)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    after = len(df)

    if after == 0:
        print("❌ No valid rows after date parsing.", file=sys.stderr)
        return 5

    keep = max(1, int(after * args.fraction))
    df_recent = df.iloc[-keep:]

    try:
        df_recent.to_csv(args.output, index=False)
    except Exception as e:
        print(f"❌ Failed to write output CSV: {e}", file=sys.stderr)
        return 6

    print(f"✅ Trimmed '{args.input}' -> '{args.output}'")
    print(f"   Original rows: {before}")
    print(f"   Parsed & sorted rows: {after}")
    print(f"   Kept most recent: {keep} rows ({args.fraction*100:.1f}%)")
    print(f"   Date range: {df_recent[date_col].iloc[0]} -> {df_recent[date_col].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
