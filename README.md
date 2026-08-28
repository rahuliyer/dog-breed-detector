# Dog breed classifier

Transfer-learning classifier for the 133-breed [Udacity dog-images dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). A frozen ImageNet ResNet-152 backbone feeds a linear head that is trained on `dogImages/`.

## Setup

```bash
# conda
conda env create -f environment.yml
conda activate dog-breed-classifier

# or pip (install a CUDA/CPU PyTorch build from https://pytorch.org first)
pip install -r requirements.txt
```

Download and unzip the dataset next to this repo (or pass `--dataset-root`):

```bash
curl -L -o dogImages.zip \
  https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip
```

Expected layout:

```
dogImages/train/<id>.<BreedName>/*.jpg
dogImages/valid/...
dogImages/test/...
```

## Usage

```bash
python dog_breed_classifier.py train
python dog_breed_classifier.py test --model best_model.pt
python dog_breed_classifier.py predict path/to/dog.jpg
```

Useful flags: `--dataset-root`, `--model`, `--batch-size`, `--epochs`, `--lr`, `--device`, `--patience`.

If the top predicted breed has probability ≥ 0.8 the script prints that breed only; otherwise it prints the top 3.

## Tests

```bash
python -m unittest test_dog_breed_classifier.py
```

Unit tests cover breed-name parsing, the confidence rule, checkpoint save/load, and the CLI. They do not download ImageNet weights or train on `dogImages/`.
