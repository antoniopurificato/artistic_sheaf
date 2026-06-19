In order to run CLIP or SigLIP:

```
python3 -m competitors.clip --dataset NAME_OF_THE_DATASET --model_type ['clip', 'siglip']
```

In order to run ColQwen or Colpali:

```
python3 -m competitors.competitors --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA --model_type ['colpali', 'colqwen2']
```            

For example:

```
python3 -m competitors.coli_approaches --dataset SemArt --base_folder data
```

If you want to run MSC:

```
python3 -m competitors.msc --mode ["train", "eval"] --dataset NAME_OF_THE_DATASET --test_data PATH_TO_TEST_DATA_JSON_FILE --base_folder FOLDER_CONTAINING_JSON 
```


For example:

```
python3 -m competitors.msc --mode train --base_folder data --dataset SemArt
```
