"""
Module 0: Preliminary Data Conversion
Convert raw Reddit data (zst/jsonl) to optimized Parquet format.
Run this before module_1 if your raw data is in zst or jsonl format.
"""

import io
import json
import logging
import polars as pl
import polars.selectors as cs
import zstandard as zstd
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

COLS_TO_DROP = [
    'media_embed', 'secure_media_embed', 'author_flair_css_class',
    'author_flair_text', 'distinguished', 'edited', 'link_flair_css_class',
    'link_flair_text', 'media', 'secure_media', 'suggested_sort',
    'thumbnail_height', 'thumbnail_width'
]


def extract_zstd(filepath, condition=None):
    """
    Processes a JSON stream from a zstd-compressed file and yields objects.

    Args:
        filepath (str): Path to the compressed JSON file
        condition (callable): Function to evaluate each object; if None, all objects are yielded

    Yields:
        dict: Each JSON object that meets the condition
    """
    count = 0
    with open(filepath, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(compressed_file) as stream_reader:
            text_content = io.TextIOWrapper(stream_reader, encoding='utf-8')
            for line in text_content:
                obj = json.loads(line)
                if condition is None or condition(obj):
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"{count} rows extracted.")
                    yield obj


def process_reddit_jsonl_to_parquet(input_file: str, output_file: str) -> pl.DataFrame:
    """
    Convert Reddit JSONL/zst to Parquet using Polars.
    Automatically handles empty structures and drops unnecessary columns.
    """
    input_path = Path(input_file)

    if input_path.suffix == '.zst':
        data: List[Dict] = list(extract_zstd(input_file))
        df = pl.DataFrame(data, infer_schema_length=None)
    elif input_path.suffix == '.jsonl':
        df = pl.read_ndjson(input_file, infer_schema_length=None)
    else:
        raise ValueError(f"Extension {input_path.suffix} not supported")

    # Drop columns that exist in the dataframe
    existing_cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if existing_cols_to_drop:
        df = df.drop(existing_cols_to_drop)

    # Remove struct columns
    df = df.select(~cs.struct())

    logger.info(f"Schema: {df.schema}")

    # Save as Parquet
    df.write_parquet(output_file, compression='snappy', use_pyarrow=False)

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"Converted {len(df)} records from {input_file} to {output_file}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    return df


def convert_all_reddit_data(raw_dir: str = "data/raw"):
    """Convert all JSONL/zst files in a directory to Parquet"""
    raw_path = Path(raw_dir)

    for source_file in list(raw_path.glob("*.jsonl")) + list(raw_path.glob("*.zst")):
        parquet_file = source_file.with_suffix('.parquet')

        logger.info(f"Processing: {source_file.name}")
        try:
            df = process_reddit_jsonl_to_parquet(str(source_file), str(parquet_file))
            logger.info(f"  Columns: {len(df.columns)}, Rows: {len(df)}")
        except Exception as e:
            logger.error(f"Error converting {source_file.name}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Single file example
    df = process_reddit_jsonl_to_parquet(
        'data/raw/litigi_submissions.zst',
        'data/raw/litigi_submissions.parquet'
    )

    # Or convert everything:
    # convert_all_reddit_data('data/raw')
