In order to run CLIP or SigLIP:

```
python3 -m competitors.clip --dataset NAME_OF_THE_DATASET --model_type ['clip', 'siglip']
```            

For example:

```
python3 -m competitors.clip --dataset SemArt --model_type clip
```

In order to run ColQwen or Colpali:

```
python3 -m competitors.competitors --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA --model_type ['colpali', 'colqwen2']
```            

For example:

```
python3 -m competitors.coli_approaches --dataset SemArt --base_folder data --model_type colpali
```

If you want to run ArtSAGENet:

```
python3 -m competitors.artsagenet --dataset NAME_OF_THE_DATASET  --task POSSIBLE_TASKS
```

For example:

```
python3 -m competitors.artsagenet --dataset SemArt --task classification
```

With `POSSIBLE_TASKS` being `["classification", "retrieval"]`

If you want to run MSC:

```
python3 -m competitors.msc --dataset NAME_OF_THE_DATASET --base_folder data 
```

For example:

```
python3 -m competitors.msc --base_folder data --dataset SemArt
```
