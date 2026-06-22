In order to run ColQwen or Colpali:

```
python3 -m competitors.coli_approaches --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA --model_type ['colpali', 'colqwen2']
```            

For example:

```
python3 -m competitors.coli_approaches --dataset SemArtPlus --base_folder data
```

If you want to run MSC:

```
python3 -m competitors.msc --mode ["train", "eval"] --dataset NAME_OF_THE_DATASET --test_data PATH_TO_TEST_DATA_JSON_FILE --base_folder FOLDER_CONTAINING_JSON 
```


For example:

```
python3 -m competitors.msc --mode train --base_folder data --dataset SemArtPlus
```

In case you want to run CLIP finetuned, you first need to finetune it:

```
python -m competitors.clip_ft \
    --dataset DATASET_NAME \
    --finetune_clip \
    --clip_model ViT-B-32 \
    --clip_pretrained laion2b_s34b_b79k \
    --clip_epochs 5 \
    --clip_lr 1e-5 \
    --clip_batch_size 128
```

In case you want to run SigLip finetuned, you first need to finetune it:

```
python competitorsd.siglip_finetune \
    --dataset DATASET_NAME \
    --finetune_siglip \
    --siglip_model ViT-SO400M-14-SigLIP \
    --siglip_pretrained webli \
    --siglip_epochs 5 \
    --siglip_lr 1e-6 \
    --siglip_batch_size 32
```

In order to run CLIP finetuned:

```
python3 -m competitors.clip_ft --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA
```            

For example:

```
python3 -m competitors.clip_ft --dataset SemArtPlus --base_folder data
```

In order to run SigLIP finetuned:

```
python3 -m competitors.siglip_ft --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA
```            

For example:

```
python3 -m competitors.siglip_ft --dataset SemArtPlus --base_folder data
```

In order to run GraphCLIP:

```
python3 -m competitors.graphclip --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA
```            

For example:

```
python3 -m competitors.graphclip --dataset SemArtPlus --base_folder data
```

In order to run RCML:

```
python3 -m competitors.rcml --dataset NAME_OF_THE_DATASET --base_folder PATH_TO_BASE_FOLDER_OF_DATA
```            

For example:

```
python3 -m competitors.rcml --dataset SemArtPlus --base_folder data
```
