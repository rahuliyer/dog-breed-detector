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

Useful flags: `--dataset-root`, `--model`, `--batch-size`, `--epochs`, `--lr`, `--device`, `--patience`, `--num-workers`, `--seed`, and `--no-amp`.

If the top predicted breed has probability ≥ 0.8 the script prints that breed only; otherwise it prints the top 3.

## Reproduced result

The locked environment was trained on September 4, 2026 using the published
Udacity split (6,680 train / 835 validation / 836 test images, 133 breeds).
Model selection used validation loss; the test split was evaluated once after
training from the saved best checkpoint.

| Setting | Value |
| --- | --- |
| Backbone | ImageNet-pretrained ResNet-152, frozen |
| Trainable layer | 133-class linear head |
| Hardware | 2× NVIDIA GeForce RTX 2080 (DataParallel) |
| Runtime | PyTorch 2.13.0+cu130, automatic mixed precision |
| Optimization | SGD, batch 64, learning rate 0.01, cosine schedule, seed 42 |
| Training | 50 epochs; best validation loss 0.2431 at epoch 45 |
| Test top-1 | **92.58%** (774/836) |
| Test top-5 | **99.88%** (835/836) |
| Test loss | 0.2486 |

Reproduce the run with:

```bash
uv run dog-breed-classifier train \
  --model outputs/best_model.pt \
  --batch-size 64 --epochs 50 --lr 0.01 \
  --num-workers 8 --patience 7 --seed 42
uv run dog-breed-classifier test \
  --model outputs/best_model.pt --batch-size 128 --num-workers 8 --seed 42
```

These figures are specific to Udacity's 133-class split. They should not be
compared directly with Stanford Dogs or Kaggle Dog Breed Identification results,
which use different images, labels, and train/test protocols.

## Training and checkpoint behavior

- CUDA training uses mixed precision by default and all visible GPUs when more
  than one is available. Pass `--no-amp` to disable mixed precision.
- The frozen backbone remains in evaluation mode during head training, so its
  BatchNorm running statistics do not drift.
- Checkpoints include the class ordering, image size, best epoch, and validation
  loss. Loading a complete checkpoint does not download ImageNet weights again.
- Train, validation, and test class-folder mappings must match. Evaluation always
  reloads the requested checkpoint, avoiding accidental scoring of the final
  epoch instead of the validation-selected model.

## Tests

```bash
uv run python -m unittest test_dog_breed_classifier.py
```

Unit tests cover breed-name parsing, the confidence rule, data/class validation,
checkpoint save/load, training/evaluation behavior, transforms, multi-device
selection, and the CLI. They do not download ImageNet weights or train on the
full `dogImages/` dataset.
