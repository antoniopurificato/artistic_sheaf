# Sheaf Neural Networks for multi-modal image retrieval

Clone the repository:

```
git clone https://github.com/antoniopurificato/artistic_sheaf.git
```

Jump into the repo:

```
cd artistic_sheaf && mkdir data
```

Download the data:

- To download SemArt visit [this site](https://researchdata.aston.ac.uk/id/eprint/380/) and download the ZIP. Then unzip it in the `data` folder.

- To download data from the Hertziana collection visit [this site](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.1GN3OL) and [this site](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.Z8W2JR). Download the 2 ZIP (one for each site) from the top right `Access this Dataset` buttons. Then unzip both of them in the data folder.

Now, run:

```
bash data/setup.sh
```

Install the necessary requirements:

```
pip install -r requirements.txt
```

If you want to run the main:

```
python3 main.py --batch_size BATCH_SIZE --lr LEARNING_RATE --sheaf_layers NUMBER_OF_SHEAF_LAYERS --dataset NAME_OF_THE_DATASET --epochs NUM_EPOCHS
```

You can run a simple example with default parameters using:

```
python3 main.py
```

If you want to test the competitors, the `competitors` folder contains a README that allows to easily run the competitors
