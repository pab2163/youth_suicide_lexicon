# youth_suicide_lexicon


Example Usage:

To make sure all requirements are installed:
```
pip install -r requirements.txt
```


OR use conda

```
conda env create -f environment.yml
conda activate youth_suicide_lexicon
```


```
python flag_script.py \
  --filepaths data/file1.csv data/file2.csv \
  --output_path data/flagged_output.csv \
  --text_column message_text \
  --keep_columns id_app timestamp
```