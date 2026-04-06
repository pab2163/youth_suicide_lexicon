# Youth Suicide Lexicon

A lexicon-based NLP tool for identifying suicide-related language in smartphone keyboard entries from adolescents. Developed and validated in Bloom, Treves et al. (2025) — see [Citation](#citation) below.

This repository provides:
- A custom youth suicide lexicon (tokens, token pairs, and celebrity/historical suicide references)
- A flagging pipeline to apply these lexicons to CSV files of text data
- Supporting code from the manuscript for benchmarking via the Swaminathan et al. (2023) and Low et al. (2024) lexicons

---


## Environment setup

**With conda (recommended):**

```bash
conda env create -f environment.yml
conda activate youth_suicide_lexicon
```

**With pip:**

```bash
pip install -r requirements.txt
```

---

## Input data format

The flagging script expects one or more CSV files where:

- Each row is a single text entry (e.g., one smartphone keyboard message)
- One column contains the text to analyze — typically pre-cleaned text from smartphone keyboard inputs (lowercased, deduplicated, etc.)
- Additional columns (e.g., participant ID, timestamp) can be retained in the output via `--keep_columns`

Example input:

| id_message | timestamp           | text_clean                        |
|------------|---------------------|-----------------------------------|
| 001        | 2024-01-15 10:32:00 | i dont want to be here anymore    |
| 002        | 2024-01-15 11:05:00 | heading to practice see you there |

Non-UTF-8 rows and duplicate rows are automatically dropped before flagging.

---

## Running the flagging pipeline

```bash
python flagging_script.py \
  --filepaths data/file1.csv data/file2.csv \
  --output_path data/flagged_output.csv \
  --text_column text_clean \
  --keep_columns id_message id_app timestamp
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--filepaths` | Yes | — | One or more paths to input CSV files |
| `--output_path` | Yes | — | Path for the combined output CSV |
| `--text_column` | No | `text_clean` | Name of the column containing text to flag |
| `--keep_columns` | No | None | Additional columns to retain in output (e.g., IDs, timestamps) |

**Outputs:**

The script writes two files:

- `flagged_output.csv` — full output including the text column
- `flagged_output_notext.csv` — same output with the text column removed (for sharing without raw text)

Multiple input files are combined into a single output. A `filepath` column is added to indicate the source file for each row.

---

## Output columns

Each output row contains the retained columns plus the following flag columns:

| Column | Description |
|---|---|
| `youth_suicide_lexicon_tokens` | 1 if the text matched any single token in the custom youth suicide lexicon |
| `youth_suicide_lexicon_celeb` | 1 if the text matched a celebrity or historical suicide reference |
| `youth_suicide_lexicon_pairs` | 1 if the text matched a token pair in the custom lexicon (e.g., a method word co-occurring with a self-referential word) |
| `suicide_lexicon_swaminathan_2023` | 1 if the text matched any token across all categories in the Swaminathan et al. (2023) lexicon |
| `suicide_lexicon_swaminathan_2023_thoughts_methods_only` | 1 if the text matched a token in the Suicidal Thoughts or Suicide Methods categories only |
| `emoji_flag` | 1 if the text contained one or more suicide-related emojis |
| `emojis_found` | Comma-separated list of matched emoji labels |
| `filepath` | Source file path for the row |

---

## Verifying your setup

A test script is included to verify the pipeline runs correctly on synthetic data:

```bash
bash test_run_script.sh
```

This runs the flagging script on the sample CSVs in `test_data/inputs/` and writes output to `test_data/outputs/`.

To run the Python unit tests:

```bash
pytest test_lexicon_functions.py -v
```

---

## Citation

If you use this lexicon or code in your work, please cite:

> Bloom, P. A., Treves, I. N., Salem, S., Durham, K., Zaccaria, V., Spence, J., ... & Auerbach, R. P. (2025). *Identifying suicide-related language in smartphone keyboard entries among high-risk adolescents*. PsyArXiv. https://doi.org/10.31234/osf.io/gfa7h_v2

The Swaminathan et al. (2023) lexicon included in this repository should be cited separately if used:

> Swaminathan, A., et al. (2023). *Natural language processing system for rapid detection and intervention of mental health crisis chat messages*. npj Digital Medicine. https://www.nature.com/articles/s41746-023-00951-3

The Low et al. (2024) lexicon included in this repository should be cited separately if used:

> Low, D., et al. (2024). *Using Generative AI to create lexicons for interpretable text models with high content validity*. OSF. https://osf.io/vf2bc