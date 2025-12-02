In order to run ColQwen or Colpali:

```
python3 -m competitors.competitors --data NAME_OF_JSON_file_with_triplets --base_folder PATH_TO_BASE_FOLDER_OF_DATA --model_type ['colpali', 'colqwen2']
```            

For example:

```
python3 -m competitors.coli_approaches --data triplets_semart_test_csv.json --base_folder data
```

If you want to run MSC:

```
python3 -m competitors.msc --mode ["train", "eval"] --data PATH_TO_TRAINING_DATA_JSON_FILE --test_data PATH_TO_TEST_DATA_JSON_FILE --base_folder FOLDER_CONTAINING_JSON 
```


For example:

```
python3 -m competitors.msc --mode train --base_folder data/
```