# Dog breed classifier

Transfer-learning classifier for the 133-breed [Udacity dog-images dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). A frozen ImageNet ResNet-152 backbone feeds a linear head that is trained on `dogImages/`.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

That creates `.venv` from `pyproject.toml` / `uv.lock`. PyPI torch wheels are CUDA-enabled on Linux and CPU-only on macOS and Windows. To force a backend with `uv pip` (for example CPU-only on Linux):

```bash
uv pip install torch torchvision --torch-backend=cpu
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
uv run dog-breed-classifier train
uv run dog-breed-classifier test --model best_model.pt
uv run dog-breed-classifier predict path/to/dog.jpg
```

Useful flags: `--dataset-root`, `--model`, `--batch-size`, `--epochs`, `--lr`, `--device`, `--patience`.

If the top predicted breed has probability ≥ 0.8 the script prints that breed only; otherwise it prints the top 3.

## Tests

```bash
uv run python -m unittest test_dog_breed_classifier.py
```

Unit tests cover breed-name parsing, the confidence rule, checkpoint save/load, and the CLI. They do not download ImageNet weights or train on `dogImages/`.
