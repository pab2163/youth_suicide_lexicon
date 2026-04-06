import pandas as pd
import re
import string
import re
import emoji
import numpy as np
from tqdm import tqdm
from io import StringIO
from pathlib import Path 

LEXICON_DIR = Path(__file__).parent / "lexicons"


def robust_read_csv(fp):
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        cleaned_text = f.read()
    return pd.read_csv(StringIO(cleaned_text))


def preproc_text(text):
    """
    Preprocesses input text by converting it to lowercase and removing punctuation,
    except for the '|' character.

    Args:
        text (str): The input string to preprocess.

    Returns:
        str: The cleaned and lowercased text with '|' preserved.
    """
    allowed_punct = '|'
    punct_to_remove = ''.join(ch for ch in string.punctuation if ch not in allowed_punct)
    text = text.lower()
    text = text.translate(str.maketrans('', '', punct_to_remove))
    return text


def load_codebook_with_pairs(df):
    """
    Loads a codebook CSV file and constructs compiled regular expression pattern pairs.

    This function reads a CSV file where each row defines a set of "means" tokens and "pairs" tokens.
    Based on whether the "freestanding_means" column is True, it optionally applies word boundaries
    to the "means" tokens. The function returns a list of tuples containing compiled regex patterns
    for matching these token pairs in text.

    Args:
        csv_path (str): Path to the CSV file containing the codebook.

    Returns:
        list of tuple: A list of tuples, each containing two compiled regex patterns 
        (means_pattern, pairs_pattern).
    """
    # If freestanding not present, assume False
    if 'freestanding_token' not in df.columns:
        df['freestanding_token'] = False

    if 'freestanding_pair' not in df.columns:
        df['freestanding_pair'] = False

    patterns = []
    for _, row in df.iterrows():
        # Choose wrapping for word boundaries
        # for 'tokens' column
        if row['freestanding_token']:
            means_stems = [fr'\b{re.escape(stem.strip())}\b' for stem in row['tokens'].split('|')]
        else:
            means_stems = [re.escape(stem.strip()) for stem in row['tokens'].split('|')]

        # for 'pairs' columm
        if row['freestanding_pair']:
            pairs_stems = [fr'\b{re.escape(stem.strip())}\b' for stem in row['pairs'].split('|')]
        else:
            pairs_stems = [re.escape(stem.strip()) for stem in row['pairs'].split('|')]

        means_pattern = '|'.join(means_stems)
        pairs_pattern = '|'.join(pairs_stems)

        patterns.append((re.compile(means_pattern, re.IGNORECASE), re.compile(pairs_pattern, re.IGNORECASE)))
    
    return patterns

def load_codebook_with_tokens(df):
    """
    Prepares a list of compiled regex patterns from a DataFrame of tokens and boundary flags.

    Args:
        df (pd.DataFrame): A DataFrame with at least the following columns:
            - 'token' (str): Token to search for.
            - 'word_start' (int): 1 if token must start on a word boundary.
            - 'word_end' (int): 1 if token must end on a word boundary.

    Returns:
        list of tuple: List of tuples (compiled_pattern, original_token) for fast matching.
    """
    patterns = []
    for _, row in df.iterrows():
        token = re.escape(row['token'])

        if row.get('word_start', 0):
            token = r'\b' + token
        if row.get('word_end', 0):
            token = token + r'\b'

        compiled = re.compile(token, flags=re.IGNORECASE)
        patterns.append((compiled, row['token']))
    
    return patterns


def check_text_against_tokens(text, patterns, debug=False):
    """
    Checks if any precompiled regex token pattern matches the input text.

    Args:
        text (str): The input string to be checked.
        patterns (list of tuple): List of (compiled_pattern, original_token).

    Returns:
        int: 1 if any token matches the text, otherwise 0.
    """
    for compiled, token in patterns:
        if compiled.search(text):
            if debug:
                print([text, token])
            return 1
    return 0

def check_text_against_pairs(text, patterns, debug=False):
    """
    Checks if any pair of compiled patterns simultaneously match the input text.

    For each pattern pair (means_pattern, pairs_pattern), returns 1 if both patterns match
    the text, otherwise returns 0. Prints the matching details for debugging if a match is found.

    Args:
        text (str): The input string to be checked.
        patterns (list of tuple): A list of tuples with compiled regex patterns.

    Returns:
        int: 1 if a pattern pair matches the text, otherwise 0.
    """
    for means_re, pairs_re in patterns:
        if means_re.search(text) and pairs_re.search(text):
            if debug:
                print([text, means_re, pairs_re])
            return 1
    return 0



def apply_codebook_to_column(df, text_column, patterns, paired, progress_label=''):
    """
    Applies pattern matching to a DataFrame column using either token pairs or single tokens.
    
    Displays a progress bar during processing.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str): Name of the column with text to check.
        patterns (list): A list of pattern pairs (tuples of regex) or single tokens (str).
        paired (bool): If True, checks for token pairs. If False, checks for single tokens.
    
    Returns:
        pd.Series: A Series of 0s and 1s indicating matches for each row in the column.
    """
    if paired:
        tqdm.pandas(desc=f"Flagging Custom Lexicon Token Pairs: {progress_label}")  # initialize tqdm for pandas
    else:
        tqdm.pandas(desc=f"Flagging Custom Lexicon Tokens: {progress_label}")  # initialize tqdm for pandas

    if paired:
        return df[text_column].progress_apply(lambda x: check_text_against_pairs(x, patterns=patterns))
    else:
        return df[text_column].progress_apply(lambda x: check_text_against_tokens(x, patterns=patterns))

        

def get_matching_token(text, patterns, paired):
    """
    Returns the first token or pattern that matches the input text.

    Args:
        text (str): The input string to check.
        patterns (list): A list of token strings (if paired=False) or regex pattern pairs (if paired=True).
        paired (bool): Whether the patterns are regex pairs (True) or simple tokens (False).

    Returns:
        str or tuple or None: The matching token (str) or pattern pair (tuple of regex) that flagged the text,
                              or None if no match is found.
    """
    if paired:
        for means_re, pairs_re in patterns:
            if means_re.search(text) and pairs_re.search(text):
                return (means_re.pattern, pairs_re.pattern)
    else:
        for compiled, token in patterns:
            if compiled.search(text):
                return token
    return None



def flag_suicide_related_emojis(df, text_column, output_flag_col='emoji_flag', output_list_col='emojis_found'):
    """
    Flags suicide-related emojis in a specified text column of a DataFrame.

    Flags individual emojis associated with suicide risk and flags a broader set
    of emojis if two or more are found. Displays a tqdm progress bar during processing.

    Args:
        df (pd.DataFrame): The input DataFrame containing text data.
        text_column (str): Name of the column containing text.
        output_flag_col (str): Name of the column to store the binary flag (1 for match, 0 for no match).
        output_list_col (str): Name of the column to store a comma-separated list of found emoji labels.

    Returns:
        pd.DataFrame: Original DataFrame with two additional columns:
                      - `output_flag_col`: 0 or 1 depending on match.
                      - `output_list_col`: comma-separated names of matched emojis.
    """
    tqdm.pandas(desc="Flagging Emojis")

    # Emojis to flag individually
    single_emoji_flags = {
        '🔫': 'gun',
        '💀': 'skull',
        '☠️': 'skull_crossbones',
        '🧷': 'safety_pin', 
        '🔗': 'link_chain',
        '🗡️': 'dagger',
        '🩸': 'blood_drop',
        '🔪': 'knife',
        '⚰️': 'coffin',
        '⛓️': 'chain',
        '🪒': 'razor',
    }

    # Emojis to flag only when 2+ are found
    combo_emoji_flags = {
        '🚬': 'cigarette',
        '🪦': 'headstone',
        '🖤': 'black_heart',
        '💔': 'broken_heart',
        '⛓️': 'chain',
        '🔫': 'gun',
        '💀': 'skull',
        '🪒': 'razor',
        '🗡️': 'dagger',
        '☠️': 'skull_crossbones',
        '🧷': 'safety_pin',
        '🔗': 'link_chain',
        '🩸': 'blood_drop',
        '🔪': 'knife',
        '⚰️': 'coffin',
        '🌑': 'new_moon',
        '🧠': 'brain',
        '☣️': 'biohazard',
        '☢️': 'radioactive',
        '️‍🔥': 'fire',
        '💊': 'pill',
        '✂️': 'scissors',
        '💣': 'bomb',
        '💥': 'collision',
        '🚫': 'prohibited',
        '🤕': 'face_with_bandage',
        '🌧️': 'cloud_with_rain'
    }

    def detect_emojis(text):
        text = str(text)
        found_single = [name for char, name in single_emoji_flags.items() if char in text]
        found_combo = [name for char, name in combo_emoji_flags.items() if char in text]
        combo_count = sum(text.count(char) > 0 for char in combo_emoji_flags)
        flagged = bool(found_single or combo_count >= 2)

        found_names = found_single.copy()
        if combo_count >= 2:
            found_names.extend(found_combo)

        return int(flagged), ', '.join(set(found_names))

    # Apply detection function row-wise with tqdm progress bar
    flags_and_lists = df[text_column].fillna('').progress_apply(detect_emojis)
    df[[output_flag_col, output_list_col]] = pd.DataFrame(flags_and_lists.tolist(), index=df.index)

    return df



def flag_lexicon_custom(input_df, text_column):
    custom_suicide_lexicon = pd.read_csv(f'{LEXICON_DIR}/youth_suicide_lexicon_tokens.csv')
    custom_suicide_tokenpairs=pd.read_csv(f'{LEXICON_DIR}/youth_suicide_lexicon_token_pairs.csv')
    celebrity_historical_suicides=pd.read_table(f'{LEXICON_DIR}/youth_suicide_lexicon_celebrity_historic.txt', header=0)

    custom_suicide_lexicon['token']=custom_suicide_lexicon['token'].astype(str).apply(lambda x: preproc_text(x))
    custom_suicide_lexicon['word_start'] = custom_suicide_lexicon['word_start'].fillna(0)
    custom_suicide_lexicon['word_end'] = custom_suicide_lexicon['word_end'].fillna(0)

    celebrity_historical_suicides['word_start']=0
    celebrity_historical_suicides['word_end']=0
    celebrity_historical_suicides['token']=celebrity_historical_suicides['token'].astype(str).apply(lambda x: preproc_text(x))
    celebrity_historical_suicide_tokens = load_codebook_with_tokens(celebrity_historical_suicides)

    custom_suicide_tokenpairs['tokens']=custom_suicide_tokenpairs['tokens'].astype(str).apply(lambda x: preproc_text(x))
    custom_suicide_tokenpairs['pairs']=custom_suicide_tokenpairs['pairs'].apply(lambda x: preproc_text(x))

    suicide_tokens = load_codebook_with_tokens(custom_suicide_lexicon)
    patterns = load_codebook_with_pairs(df=custom_suicide_tokenpairs)

    df = input_df.copy()
    df[text_column]=df[text_column].astype(str)
    df[text_column]=df[text_column].apply(lambda x: preproc_text(x))

    input_df['youth_suicide_lexicon_tokens']=apply_codebook_to_column(df=df, 
                                                     text_column=text_column, 
                                                     patterns=suicide_tokens, 
                                                     paired=False,
                                                     progress_label='Single Suicide Lexicon Tokens')
    
    input_df['youth_suicide_lexicon_celeb']=apply_codebook_to_column(df=df, 
                                                     text_column=text_column, 
                                                     patterns=celebrity_historical_suicide_tokens, 
                                                     paired=False,
                                                     progress_label='Celebrity/Historical Suicides')
    
    input_df['youth_suicide_lexicon_pairs']=apply_codebook_to_column(df=df, 
                                                     text_column=text_column, 
                                                     patterns=patterns, 
                                                     paired=True,
                                                     progress_label = 'Suicide Lexicon Token Pairs')
        
    return input_df


def flag_lexicon_swaminathan_2023(input_df, text_column, debug=False):
    # Ensure text column is string and preprocessed
    df = input_df.copy()
    df[text_column] = df[text_column].astype(str).apply(lambda x: preproc_text(x))

    # Load lexicon
    swaminathan_2023_lexicon = pd.read_csv(f'{LEXICON_DIR}/Swaminathan_2023_lexicon.csv')

    # Melt full lexicon for general flag
    full_melted = swaminathan_2023_lexicon[[
        'Suicidal Thoughts', 'Suicide Methods',
        'Alcohol and Illicit Alcohol & Illicit Substances', 'Sleep',
        'Help-Seeking', 'Hopeless', 'General Risk'
    ]].melt(value_name='token')['token'].dropna().astype(str)

    full_tokens = pd.DataFrame({'token': full_melted, 'word_start': 0, 'word_end': 0})
    full_tokens['token'] = full_tokens['token'].apply(lambda x: preproc_text(x))

    # Melt subset for Suicidal Thoughts and Methods flag
    suicide_melted = swaminathan_2023_lexicon[['Suicidal Thoughts', 'Suicide Methods']].melt(value_name='token')['token'].dropna().astype(str)

    suicide_tokens = pd.DataFrame({'token': suicide_melted, 'word_start': 0, 'word_end': 0})
    suicide_tokens['token'] = suicide_tokens['token'].apply(lambda x: preproc_text(x))

    # Shared boundary adjustment
    boundary_terms = ['od', 'gun', 'cry', 'hang', 'burn', 'so alone', 'hate', 'weep', 'ugh', 'meth', 'die']
    full_tokens.loc[full_tokens['token'].isin(boundary_terms), 'word_start'] = 1
    suicide_tokens.loc[suicide_tokens['token'].isin(boundary_terms), 'word_start'] = 1

    # Compile regex patterns
    full_patterns = load_codebook_with_tokens(full_tokens)
    suicide_patterns = load_codebook_with_tokens(suicide_tokens)

    # Apply both sets
    input_df['suicide_lexicon_swaminathan_2023'] = apply_codebook_to_column(
        df=df,
        text_column=text_column,
        patterns=full_patterns,
        paired=False,
        progress_label='Swaminathan_2023_lexicon'
    )

    input_df['suicide_lexicon_swaminathan_2023_thoughts_methods_only'] = apply_codebook_to_column(
        df=df,
        text_column=text_column,
        patterns=suicide_patterns,
        paired=False,
        progress_label='Swaminathan_2023_lexicon_subset'
    )

    # Debug info
    if debug:
        for pattern, original in suicide_patterns:
            matches = df[df[text_column].str.contains(pattern.pattern, case=False, regex=True)]
            if not matches.empty:
                print(f"Subset Match Pattern: {original}")
                print(matches[[text_column]])

    return input_df