# youth_suicide_lexicon


## Environment setup

Probably the simplest way to set up the environment is via conda and the `environment.yml` file inside this folder. Make sure you are in the top level of the directory where `environment.yml` is, then run the following to use it to set up a conda environment with the required packages

```
conda env create -f environment.yml
```

Then, run the following to activate the conda environment

```
conda activate youth_suicide_lexicon
```


### Install without conda 

For install without conda, requirements can be install via pip with the following
```
pip install -r requirements.txt
```


## To flag data with this lexicon:

```
python flagging_script.py \
  --filepaths data/file1.csv data/file2.csv \
  --output_path data/flagged_output.csv \
  --text_column message_text \
  --keep_columns id_app timestamp
```

Arguments

`filepaths`: list of files to be flagged
`output_path`: base path to output flagged data (will create 2 versions, one without the text)
`text_column`: the column in the input files you want used to flag
`keep_columns`: list of any other columns in the csv to retain (subjectID, timestamps, app, etc)