# Art Beyond Semantics: Sheaf-Informed Contrastive Learning for Multi-Relational Representations

Experiments tested on Python 3.10.

Clone the repo:

```
https://github.com/antoniopurificato/artistic_sheaf.git
```

Jump into the repo:

```
cd artistic_sheaf
```

Install the necessary requirements:

```
pip install -r requirements.txt
```

If you want to run the main:

```
python3 main.py --batch_size BATCH_SIZE --lr LEARNING_RATE --sheaf_layers NUMBER_OF_SHEAF_LAYERS --dataset DATASET_NAME --epochs NUM_EPOCHS
```

You can run a simple example with default parameters using:

```
python3 main.py
```

If you want to run a sweep:

```
python3 main.py --sweep True --params_sweep POSSIBLE_SWEEP_CHOICES --dataset DATASET_NAME --project_name_wandb PROJECT_NAME_IN_WANDB --entity_name_wandb ENTITY_NAME_IN_WANDB
```

Example:

```
python3 main.py --sweep True --params_sweep batch_size --dataset HertzianaDP
```

`POSSIBLE_SWEEP_CHOICES` could be `batch_size, lr, sheaf_layers, finetune_layers`. You can select the values in `params.yaml`

Datasets could be `SemArtPlus`, `HertzianaDP` or `WikiArtPlus`.

If you want to test the competitors, the `competitors` folder contains a README that allows to easily run the competitors.

For the `HertzianaDP` dataset, we do not have the corresponding HuggingFace datasets due to permissions. If you want to download the data, refer to [this link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.Z8W2JR) and [this link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.1GN3OL). If you want to test CANVAS on HertzianaDP contact us and we will help you.

If you use our code, please cite the associated paper:
```
@misc{schaerf2026artsemanticssheafinformedcontrastive,
      title={Art Beyond Semantics: Sheaf-Informed Contrastive Learning for Multi-Relational Representations}, 
      author={Ludovica Schaerf and Antonio Purificato and Piera Riccio and Fabrizio Silvestri and Noa Garcia},
      year={2026},
      eprint={2607.16321},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.16321}, 
}
```
