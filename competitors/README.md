In order to run ColQwen or Colpali:

```
python3 -m competitors.competitors --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA --model_type ['colpali', 'colqwen2']
```            

For example:

```
python3 -m competitors.coli_approaches --dataset SemArt --base_folder data
```

If you want to run ArtSAGENET:

```
python3 -m competitors.artsagenet --dataset NAME_OF_THE_DATASET  --task POSSIBLE_TASKS
```

For example:

```
python3 -m competitors.artsagenet --dataset SemArt --task classification

With `POSSIBLE_TASKS` being `["classification", "regression", "retrieval"]`

If you want to run MSC:

```
python3 -m competitors.msc --dataset NAME_OF_THE_DATASET --base_folder data 
```


For example:

```
python3 -m competitors.msc --base_folder data --dataset SemArt
```


If you want to run EKG:

```
python3 -m competitors.ekg --dataset NAME_OF_THE_DATASET --base_folder data 
```


For example:

```
python3 -m competitors.ekg --base_folder data --dataset SemArt
```