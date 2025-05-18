python flagging_script.py \
  --filepaths test_data/inputs/fake_data1.csv test_data/inputs/fake_data2.csv test_data/inputs/fake_data3.csv \
  --output_path test_data/outputs/flagged_output.csv \
  --text_column text_preproc \
  --keep_columns id_message id_app id_row tm_message_start tm_message_end flagged