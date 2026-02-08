import polars as pl
import jsonlines
from pathlib import Path
from typing import List, Dict
import zstandard as zstd
import json, io
import polars.selectors as cs

def extract_zstd(filepath,condition=None):
    """
    Processes a JSON stream from a compressed file and yields objects based on condition.

    Args:
        filepath (str): Path to the compressed JSON file
        condition (callable): Function to evaluate each object; if None, all objects are yielded

    Yields:
        dict: Each JSON object that meets the condition
    """
    i=0
    with open(filepath, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(compressed_file) as stream_reader:
            # Read all content into a buffer
            text_content = io.TextIOWrapper(stream_reader, encoding='utf-8')
            for line in text_content:
                obj = json.loads(line)
                if condition is None or condition(obj):
                    i=i+1
                    if i%1000==0:
                        print (i, ' rows extracted.')
                    yield obj


def process_reddit_jsonl_to_parquet(input_file: str, output_file: str):
    """
    Converte Reddit JSONL in Parquet usando Polars
    Gestisce automaticamente le strutture vuote
    """
    cols_to_drop = ['media_embed','secure_media_embed','author_flair_css_class', 'author_flair_text', 'distinguished', 'edited', 'link_flair_css_class', 'link_flair_text', 'media', 'secure_media', 'suggested_sort', 'thumbnail_height', 'thumbnail_width']
    if Path(input_file).suffix == '.zst':
        data :List[Dict] = [entry for entry in extract_zstd(input_file)]
        df =pl.DataFrame(data,infer_schema_length=None).drop(cols_to_drop)
    elif Path(input_file).suffix == '.jsonl':
        # Polars legge NDJSON (newline-delimited JSON) direttamente
        df = pl.read_ndjson(input_file,infer_schema_length=None).drop(cols_to_drop)
    else:
        raise ValueError(f"Extension {Path(input_file).suffix} not supported")
    print(df.columns)
    print(df.head(10))
    
    print([c for c in df.columns if 'media_embed' in c])
    df = df.select(~cs.struct())
    print(df.schema)
    # Salva come Parquet con compressione
    cols_to_write = ['author','subreddit','num_comments','num_crossposts','id','created_utc','archived','body']
    df.select(cols_to_write).write_parquet(
        output_file,
        compression='snappy',
        use_pyarrow=False  # Usa engine nativo Polars
    )
    
    print(f"✓ Convertiti {len(df)} record da {input_file} a {output_file}")
    print(f"  Dimensione file: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
    return df

# Batch conversion per tutti i file JSONL
def convert_all_reddit_data(raw_dir: str = "data/raw"):
    """
    Converte tutti i file JSONL in Parquet
    """
    raw_path = Path(raw_dir)
    
    for jsonl_file in raw_path.glob("*.jsonl"):
        parquet_file = jsonl_file.with_suffix('.parquet')
        
        print(f"\nProcessing: {jsonl_file.name}")
        try:
            df = process_reddit_jsonl_to_parquet(str(jsonl_file), str(parquet_file))
            print(f"  Columns: {len(df.columns)}, Rows: {len(df)}")
        except Exception as e:
            print(f"❌ Error: {e}")

# Per il tuo caso specifico
if __name__ == "__main__":
    # Singolo file
    df = process_reddit_jsonl_to_parquet(
        'data/raw/litigi_submissions.zst',
        'data/raw/litigi_submissions.parquet'
    )
    
    # O converti tutto
    # convert_all_reddit_data('data/raw')