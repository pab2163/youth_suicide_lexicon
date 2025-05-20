import pickle
import sys
from sentence_transformers import SentenceTransformer
import os
import sys
import shutil
import datetime
import copy
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from lexicon_functions import robust_read_csv

# Append source directories to path
sys.path.append('./../src/')
sys.path.append('./../src/construct_tracker/')

# Import construct_tracker modules
from construct_tracker import lexicon
from construct_tracker import cts
from sentence_transformers import SentenceTransformer
'''
A blank pickle file is needed for the step of getting cosine similarities between input text and lexicon constructs

'''

# Define an empty object (e.g., an empty list or dictionary)
empty_object = {}  # Or use {} for an empty dictionary, etc.

# Specify the file name for your pickle file
if not os.path.isfile('data/embeddings/stored_embeddings.pickle'):
    os.system('mkdir data')
    os.system('mkdir data/embeddings')
    file_path = 'data/embeddings/stored_embeddings.pickle'

    # Open the file in binary write mode
    with open(file_path, 'wb') as file:
        pickle.dump(empty_object, file)

    print(f"Blank pickle file created: {file_path}")
else:
    print(f'Using stored embeddings in data/embeddings/stored_embeddings.pickle')

'''
Download the model locally to ensure this works without data touching the cloud
'''

if not os.path.isfile('models/all-MiniLM-L6-v2-local/config.json'):
    os.system('mkdir models')
    modelPath = 'models/all-MiniLM-L6-v2-local'
    print('Downloading and saving model all-MiniLM-L6-v2 to path models/all-MiniLM-L6-v2-local')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(modelPath)

else:
    print('Using local transformer model at: models/all-MiniLM-L6-v2-local')


# ---------------------------
# Load Lexicons
# ---------------------------

try:
    # Load full SRL lexicon
    srl = lexicon.load_lexicon(name='srl_v1-0')

    # Load prototype tokens only
    srl_prototypes = lexicon.load_lexicon(name='srl_prototypes_v1-0')
except Exception as e:
    raise RuntimeError(f"Error loading lexicons: {e}")

lexicon_dict = {
    c: srl_prototypes.constructs[c]["tokens"]
    for c in srl_prototypes.constructs
}


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

    Parameters:
    - filepaths: list of str or Path
        List of paths to input CSV files to process.
    - output_dir: str or Path
        Directory where the output files will be saved.
    - text_column: str
        Name of the column in each CSV that contains the text.
    - keep_columns: list of str or None
        List of columns to retain from the original CSV.
    - lexicon: dict
        Dictionary containing construct names mapped to their prototype tokens.
    - embeddings_model: str
        Path to sentence transformer model.
    - stored_embeddings_path: str
        Path to store or load precomputed embeddings.

    Returns:
    - List of output file paths generated.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    for fp in filepaths:
        fp = Path(fp)
        try:
            df = robust_read_csv(fp).drop_duplicates()

            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in file: {fp}")

            # Remove rows with non-UTF-8 encodable text
            def is_utf8_encodable(val):
                try:
                    val.encode('utf-8')
                    return True
                except Exception:
                    return False

            df = df[df[text_column].apply(lambda x: isinstance(x, str) and is_utf8_encodable(x))]

        except FileNotFoundError:
            warnings.warn(f"File not found: {fp}. Skipping.")
            continue
        except Exception as e:
            warnings.warn(f"Error reading file {fp}: {e}. Skipping.")
            continue

        if text_column not in df.columns:
            warnings.warn(f"Missing '{text_column}' in {fp}. Skipping.")
            continue

        if keep_columns:
            base_cols = [text_column] + [col for col in keep_columns if col in df.columns]
            df = df[base_cols]

        text_inputs = list(df[text_column].astype(str))
        if not text_inputs or all([not isinstance(t, str) or t.strip() == '' for t in text_inputs]):
            warnings.warn(f"No valid text inputs in {fp}. Skipping.")
            continue

        # ----------------------------------
        # Exact Lexicon Match Feature Extraction
        # ----------------------------------
        try:
            counts, matches_by_construct, matches_doc2construct, matches_construct2doc = srl.extract(
                text_inputs, normalize=False)
            df_counts = pd.concat([df, counts], axis=1)
            counts_path = output_dir / f"{fp.stem}_srl_lexicon_counts.csv"
            df_counts.to_csv(counts_path, index=False)
            output_files.append(counts_path)
        except Exception as e:
            warnings.warn(f"Lexicon extraction error in {fp}: {e}. Skipping.")
            continue

        # ----------------------------------
        # Cosine Similarity Features
        # ----------------------------------
        if not lexicon:
            warnings.warn(f"No lexicon provided for cosine similarity. Skipping {fp}.")
            continue

        try:
            features, lexicon_dict_final_order, cosine_similarities = cts.measure(
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
                warnings.warn(f"No cosine features computed in {fp}.")
                continue

            df_features = pd.concat([df, features], axis=1)
            features_path = output_dir / f"{fp.stem}_srl_lexicon_cts_features_sentence.csv"
            df_features.to_csv(features_path, index=False)
            output_files.append(features_path)
        except Exception as e:
            warnings.warn(f"Cosine similarity error in {fp}: {e}. Skipping.")
            continue

    return output_files


# Run extraction
extract_low_2024_srl_features_from_files(
    filepaths=['test_data/inputs/fake_data1.csv', 'test_data/inputs/fake_data2.csv', 'test_data/inputs/fake_data3.csv'],
    output_dir='test_data/outputs/low_2024',
    text_column='text_preproc',
    keep_columns=['id_message', 'id_app', 'id_row', 'tm_message_start', 'tm_message_end', 'flagged'],
)