"""
Module 0: Preliminary Data Conversion
Convert raw Reddit data (zst/jsonl) to optimized Parquet format.
Run this before module_1 if your raw data is in zst or jsonl format.

Reddit JSONL has inconsistent types across rows (e.g. `edited` is `false`
or a float timestamp), so we read line-by-line with Python's json module
to let Python normalize the types before building a Polars DataFrame.

Memory strategy: strip each row to only the needed columns at read time,
then write batches incrementally to Parquet via PyArrow's ParquetWriter
so we never hold the full dataset in memory.
"""

import io
import json
import logging
import polars as pl
import pyarrow.parquet as pq

import zstandard as zstd
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# Only keep columns actually used by the pipeline (extraction, preprocessing,
# classifier, network graph, bot). Everything else is dropped to save memory.
#
# Split by record type so each parquet file contains exactly the columns it
# needs, making downstream expectations explicit.

_COMMON_COLS = {
    'id',           # unique identifier
    'author',       # filter by / label by user
    'score',        # quality filtering
    'created_utc',  # chronological ordering (cast to datetime after read)
    'subreddit',    # subreddit name (useful for multi-sub analysis)
    'name',         # fallback identifier (used if 'id' missing)
}

COMMENT_COLS = _COMMON_COLS | {
    'parent_id',    # thread building (t1_<id> format)
    'link_id',      # maps comment to its submission (t3_<id> format)
    'body',         # comment text
}

SUBMISSION_COLS = _COMMON_COLS | {
    'selftext',     # submission (post) body text
    'title',        # submission title
}

# Union kept for the generic / auto-detect path
COLS_TO_KEEP = COMMENT_COLS | SUBMISSION_COLS

BATCH_SIZE = 50_000


def _strip_row(row: dict, cols: set[str] = COLS_TO_KEEP) -> dict:
    """Keep only *cols* keys from a parsed JSON row."""
    return {k: v for k, v in row.items() if k in cols}


def _iter_jsonl(filepath: str, cols: set[str] = COLS_TO_KEEP) -> Generator[dict, None, None]:
    """Yield stripped JSON objects from a plain JSONL file, skipping bad lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _strip_row(json.loads(line), cols)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")


def _iter_zstd(filepath: str, condition=None, cols: set[str] = COLS_TO_KEEP) -> Generator[dict, None, None]:
    """Yield stripped JSON objects from a zstd-compressed JSONL file."""
    count = 0
    with open(filepath, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(compressed_file) as stream_reader:
            text_content = io.TextIOWrapper(stream_reader, encoding='utf-8')
            for i, line in enumerate(text_content, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {i}: {e}")
                    continue
                if condition is None or condition(obj):
                    count += 1
                    if count % 10_000 == 0:
                        logger.info(f"{count} rows extracted.")
                    yield _strip_row(obj, cols)


def process_reddit_jsonl_to_parquet(
    input_file: str,
    output_file: str,
    condition=None,
    batch_size: int = BATCH_SIZE,
    cols: set[str] | None = None,
) -> pl.DataFrame:
    """
    Convert Reddit JSONL/zst to Parquet using Polars.

    Reads line-by-line via Python's json module to handle Reddit's
    inconsistent types (e.g. `edited` being bool or float).
    Each row is stripped to *cols* at read time, and batches
    are written incrementally to Parquet via PyArrow so we never
    hold the full dataset in memory.

    Pass ``cols=COMMENT_COLS`` or ``cols=SUBMISSION_COLS`` for
    type-specific column sets; defaults to the full COLS_TO_KEEP union.

    ``created_utc`` is cast from a Unix-epoch string to a proper
    ``Datetime`` column in the output Parquet.
    """
    if cols is None:
        cols = COLS_TO_KEEP

    input_path = Path(input_file)

    if input_path.suffix == '.zst':
        row_iter = _iter_zstd(input_file, condition=condition, cols=cols)
    elif input_path.suffix == '.jsonl':
        row_iter = _iter_jsonl(input_file, cols=cols)
    else:
        raise ValueError(f"Extension {input_path.suffix} not supported. Use .jsonl or .zst")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Fixed schema: all kept columns as Utf8, so every batch matches.
    sorted_cols = sorted(cols)
    fixed_schema = pl.Schema({col: pl.Utf8 for col in sorted_cols})

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    batch: list[dict] = []

    def _flush_batch(batch: list[dict], writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        # Ensure every row has all columns (None for missing)
        for row in batch:
            for col in sorted_cols:
                if col not in row:
                    row[col] = None
                elif row[col] is not None:
                    row[col] = str(row[col])
        df = pl.DataFrame(batch, schema=fixed_schema)

        # Cast created_utc from string epoch to Datetime
        if 'created_utc' in df.columns:
            df = df.with_columns(
                pl.col('created_utc')
                .cast(pl.Float64, strict=False)
                .cast(pl.Datetime('ms'))
                .alias('created_utc')
            )

        table = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
        writer.write_table(table)
        return writer

    try:
        for row in row_iter:
            batch.append(row)
            if len(batch) >= batch_size:
                writer = _flush_batch(batch, writer)
                total_rows += len(batch)
                logger.info(f"Written batch ({len(batch)} rows, {total_rows} total)")
                batch = []

        if batch:
            writer = _flush_batch(batch, writer)
            total_rows += len(batch)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise ValueError(f"No valid rows found in {input_file}")

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"Saved {total_rows} records to {output_file} ({file_size_mb:.2f} MB)")

    # Return a lightweight reference - read back from parquet (lazy scan, no full load)
    return pl.scan_parquet(output_file).head(5).collect()


def _infer_cols(filename: str) -> set[str]:
    """Pick COMMENT_COLS or SUBMISSION_COLS based on the filename."""
    name = filename.lower()
    if 'comment' in name:
        return COMMENT_COLS
    if 'submission' in name or 'post' in name:
        return SUBMISSION_COLS
    return COLS_TO_KEEP  # fallback: keep everything


def convert_all_reddit_data(raw_dir: str = "data/raw"):
    """Convert all JSONL/zst files in a directory to Parquet."""
    raw_path = Path(raw_dir)
    files = list(raw_path.glob("*.jsonl")) + list(raw_path.glob("*.zst"))

    if not files:
        logger.warning(f"No .jsonl or .zst files found in {raw_dir}")
        return

    for source_file in files:
        parquet_file = source_file.with_suffix('.parquet')
        cols = _infer_cols(source_file.name)
        logger.info(f"Processing: {source_file.name} (cols={sorted(cols)})")
        try:
            preview = process_reddit_jsonl_to_parquet(
                str(source_file), str(parquet_file), cols=cols,
            )
            logger.info(f"  Columns: {preview.columns}")
        except Exception as e:
            logger.error(f"Error converting {source_file.name}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Single file example (explicit column set)
    # df = process_reddit_jsonl_to_parquet(
    #     'data/raw/litigi_comments.jsonl',
    #     'data/raw/litigi_comments.parquet',
    #     cols=COMMENT_COLS,
    # )

    # Or convert everything (auto-detects cols from filename):
    convert_all_reddit_data('data/raw')
