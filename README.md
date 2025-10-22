# Sheaf Neural Networks for multi-modal image retrieval

Clone the repository:

```
git clone https://github.com/antoniopurificato/artistic_sheaf.git
```

Jump into the repo and create the `data` folder:

```
cd artistic_sheaf && mkdir data
```

Download the SemArt dataset:

```
cd data && wget https://researchdata.aston.ac.uk/id/eprint/380/1/SemArt.zip && cd ..
```

Then copy the json file into the `data` folder.

Install the necessary requirements:

```
pip install -r requirements.txt
```

If you want to see the shape of the data and info about them:

```
python3 -m src.data
```

If you want to run the main:

```
python3 main.py
```
