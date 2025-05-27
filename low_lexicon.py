import pickle
import sys
from sentence_transformers import SentenceTransformer
import os
import shutil
import datetime
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from lexicon_functions import robust_read_csv

# Append source directories to path
sys.path.append('./../src/')
sys.path.append('./../src/construct_tracker/')

from construct_tracker import lexicon
from construct_tracker import cts


def extract_low_2024_srl_features_from_files(
    filepaths,
    output_dir,
    text_column='text_clean',
    keep_columns=None,
    embeddings_model='models/all-MiniLM-L6-v2-local',
    stored_embeddings_path='data/embeddings/stored_embeddings.pickle'
):
    """
    Extracts suicide-related lexicon match features and cosine similarity features from a list of CSV files.

    Returns:
    - List of output file paths generated.
    """

    # Ensure blank pickle file for embeddings exists
    if not os.path.isfile(stored_embeddings_path):
        os.makedirs(Path(stored_embeddings_path).parent, exist_ok=True)
        with open(stored_embeddings_path, 'wb') as file:
            pickle.dump({}, file)
        print(f"Blank pickle file created: {stored_embeddings_path}")
    else:
        print(f'Using stored embeddings: {stored_embeddings_path}')

    # Ensure transformer model is downloaded
    if not os.path.isfile(f'{embeddings_model}/config.json'):
        os.makedirs(embeddings_model, exist_ok=True)
        print(f'Downloading model to {embeddings_model}')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(embeddings_model)
    else:
        print(f'Using local transformer model at: {embeddings_model}')

    # Load lexicons
    try:
        srl = lexicon.load_lexicon(name='srl_v1-0')
        srl_prototypes = lexicon.load_lexicon(name='srl_prototypes_v1-0')
    except Exception as e:
        raise RuntimeError(f"Error loading lexicons: {e}")

    lexicon_dict = {
        c: srl_prototypes.constructs[c]["tokens"]
        for c in srl_prototypes.constructs
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    def is_utf8_encodable(val):
        try:
            val.encode('utf-8')
            return True
        except Exception:
            return False

    for fp in filepaths:
        fp = Path(fp)
        try:
            df = robust_read_csv(fp).drop_duplicates()

            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in file: {fp}")

            df[text_column] = df[text_column].astype(str)
            valid_mask = df[text_column].apply(lambda x: x.strip() != "" and is_utf8_encodable(x))
            df_valid = df[valid_mask].copy()
            text_inputs = list(df_valid[text_column])

            if not text_inputs:
                warnings.warn(f"No valid text inputs in {fp}. Skipping.")
                continue

        except Exception as e:
            warnings.warn(f"Error reading file {fp}: {e}. Skipping.")
            continue

        # Subset columns to keep
        if keep_columns:
            cols_to_keep = [text_column] + [col for col in keep_columns if col in df.columns]
            df = df[cols_to_keep]
            df_valid = df_valid[cols_to_keep]

        # ---- SRL Lexicon Extraction ----
        try:
            counts, _, _, _ = srl.extract(text_inputs, normalize=False)
            counts.index = df_valid.index  # align with original index
            full_counts = df.join(counts, how='left')
            counts_path = output_dir / f"{fp.stem}_srl_lexicon_counts.csv"
            full_counts.to_csv(counts_path, index=False)
            output_files.append(counts_path)
        except Exception as e:
            warnings.warn(f"Lexicon extraction error in {fp}: {e}. Skipping.")
            continue

        # ---- Cosine Similarity Features ----
        try:
            features, _, _ = cts.measure(
                lexicon_dict,
                text_inputs,
                count_if_exact_match=False,
                summary_stat=['max', 'mean'],
                embeddings_model=embeddings_model,
                stored_embeddings_path=stored_embeddings_path,
                save_lexicon_embeddings=True,
                verbose=True,
                document_representation="sentence"
            )

            if features is None or features.empty:
                warnings.warn(f"No cosine features computed in {fp}. Skipping.")
                continue

            features.index = df_valid.index  # realign with filtered DataFrame
            full_features = df.join(features, how='left')
            features_path = output_dir / f"{fp.stem}_srl_lexicon_cts_features_sentence.csv"
            full_features.to_csv(features_path, index=False)
            output_files.append(features_path)

        except Exception as e:
            warnings.warn(f"Cosine similarity error in {fp}: {e}. Skipping.")
            continue

    return output_files
