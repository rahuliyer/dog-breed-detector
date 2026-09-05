"""Transfer-learning dog breed classifier using a frozen ResNet-152 backbone.

Expects the Udacity-style `dogImages/` layout:

    dogImages/train/<id>.<BreedName>/*.jpg
    dogImages/valid/...
    dogImages/test/...
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet152

# Some JPEGs in public dog-breed sets are truncated; allow PIL to load them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_PATH = "best_model.pt"
DEFAULT_DATASET_ROOT = "dogImages"
DEFAULT_NUM_BREEDS = 133
CONFIDENT_THRESHOLD = 0.8

Prediction = Tuple[str, float]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_breed_name(folder_name: str) -> str:
    """Convert ImageFolder names like '001.Affenpinscher' to 'Affenpinscher'."""
    name = folder_name.split(".", 1)[1] if "." in folder_name else folder_name
    return name.replace("_", " ")


def select_predictions(
    breed_names: Sequence[str],
    probabilities: Sequence[float],
    threshold: float = CONFIDENT_THRESHOLD,
) -> List[Prediction]:
    """Return top-1 if confident, otherwise the top-3 candidates."""
    ranked = list(zip(breed_names, probabilities))
    if not ranked:
        return []
    if ranked[0][1] >= threshold:
        return ranked[:1]
    return ranked[:3]


def build_resnet152(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        from torchvision.models import ResNet152_Weights

        weights = ResNet152_Weights.DEFAULT if pretrained else None
        model = resnet152(weights=weights)
    except (ImportError, AttributeError, TypeError):
        model = resnet152(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")
    return model


def _unwrap_state_dict(state_dict: dict) -> dict:
    """Strip a DataParallel `module.` prefix if present."""
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _torch_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class DogBreedClassifier:
    def __init__(
        self,
        dataset_root: str = DEFAULT_DATASET_ROOT,
        num_breeds: int = DEFAULT_NUM_BREEDS,
        image_size: int = 224,
        batch_size: int = 32,
        lr: float = 0.01,
        num_epochs: int = 50,
        num_workers: Optional[int] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = CONFIDENT_THRESHOLD,
        seed: int = 42,
        amp: bool = True,
    ):
        self.dataset_root = dataset_root
        self.num_breeds = num_breeds
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_workers = min(4, os.cpu_count() or 0) if num_workers is None else num_workers
        self.device = device or get_device()
        self.confidence_threshold = confidence_threshold
        self.seed = seed
        self.amp = amp and self.device.type == "cuda"

        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.best_valid_loss: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.last_test_top5: Optional[float] = None

        self._configure_transforms()

    def _configure_transforms(self) -> None:
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 256 / 224)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        # Back-compat alias used by predict_breed / older call sites.
        self.transforms = self.eval_transforms

    def get_loader(self, datadir: str, train: bool = False) -> DataLoader:
        transform = self.train_transforms if train else self.eval_transforms
        ds = ImageFolder(datadir, transform)
        if self.class_names and ds.classes != self.class_names:
            raise ValueError(
                f"Class folders in {datadir} do not match the model: "
                f"expected {self.class_names}, found {ds.classes}"
            )
        if not self.class_names:
            self.class_names = ds.classes
            self.num_breeds = len(ds.classes)
        pin_memory = self.device.type == "cuda"
        generator = torch.Generator().manual_seed(self.seed) if train else None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
            generator=generator,
        )

    def get_train_loader(self) -> DataLoader:
        return self.get_loader(os.path.join(self.dataset_root, "train"), train=True)

    def get_valid_loader(self) -> DataLoader:
        return self.get_loader(os.path.join(self.dataset_root, "valid"), train=False)

    def get_test_loader(self) -> DataLoader:
        return self.get_loader(os.path.join(self.dataset_root, "test"), train=False)

    def get_model(self, pretrained: bool = True) -> nn.Module:
        model = build_resnet152(self.num_breeds, pretrained=pretrained)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)
        return self.model

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call get_model() or load_model_from_file().")
        return self.model

    def save_model(self, filename: str = DEFAULT_MODEL_PATH) -> None:
        model = self._require_model()
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        state_dict = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        torch.save(
            {
                "state_dict": state_dict,
                "class_names": self.class_names,
                "num_breeds": self.num_breeds,
                "image_size": self.image_size,
                "best_valid_loss": self.best_valid_loss,
                "best_epoch": self.best_epoch,
            },
            filename,
        )

    def load_model(self, filename: str = DEFAULT_MODEL_PATH) -> None:
        model = self._require_model()
        state_dict = self._load_checkpoint(filename)
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(state_dict)

    def load_model_from_file(self, filename: str = DEFAULT_MODEL_PATH) -> None:
        state_dict = self._load_checkpoint(filename)
        self.get_model(pretrained=False)
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        target.load_state_dict(state_dict)

    def _load_checkpoint(self, filename: str) -> dict:
        checkpoint = _torch_load(filename, self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if checkpoint.get("class_names"):
                self.class_names = list(checkpoint["class_names"])
            if checkpoint.get("num_breeds"):
                self.num_breeds = int(checkpoint["num_breeds"])
            if checkpoint.get("image_size"):
                self.image_size = int(checkpoint["image_size"])
                self._configure_transforms()
            if checkpoint.get("best_valid_loss") is not None:
                self.best_valid_loss = float(checkpoint["best_valid_loss"])
            if checkpoint.get("best_epoch") is not None:
                self.best_epoch = int(checkpoint["best_epoch"])
            return _unwrap_state_dict(checkpoint["state_dict"])
        return _unwrap_state_dict(checkpoint)

    def _run_eval(self, loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float, float]:
        model = self._require_model()
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_top5 = 0
        total_examples = 0
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    logits = model(inputs)
                    loss = loss_fn(logits, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                top5_indices = logits.topk(min(5, logits.size(1)), dim=1).indices
                total_top5 += top5_indices.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            return 0.0, 0.0, 0.0
        return (
            total_loss / total_examples,
            total_correct / total_examples,
            total_top5 / total_examples,
        )

    def train(self, model_filename: str = DEFAULT_MODEL_PATH, patience: int = 7) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        train_loader = self.get_train_loader()
        valid_loader = self.get_valid_loader()
        self.get_model()
        model = self._require_model()
        model.train()

        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        loss_fn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        best_valid_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            model.train()
            backbone = model.module if isinstance(model, nn.DataParallel) else model
            for child in backbone.children():
                if child is not backbone.fc:
                    child.eval()
            running_loss = 0.0
            running_correct = 0
            running_examples = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    logits = model(inputs)
                    loss = loss_fn(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                running_correct += (logits.argmax(dim=1) == targets).sum().item()
                running_examples += batch_size

            train_loss = running_loss / max(running_examples, 1)
            train_acc = running_correct / max(running_examples, 1)
            valid_loss, valid_acc, _ = self._run_eval(valid_loader, loss_fn)
            scheduler.step()

            print(
                "Epoch {epoch}/{total}: "
                "train loss={train_loss:.4f} acc={train_acc:.3f} | "
                "valid loss={valid_loss:.4f} acc={valid_acc:.3f}".format(
                    epoch=epoch + 1,
                    total=self.num_epochs,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    valid_loss=valid_loss,
                    valid_acc=valid_acc,
                )
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_valid_loss = valid_loss
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
                print("Lowest validation loss; saving model to {}...".format(model_filename))
                self.save_model(model_filename)
            else:
                epochs_without_improvement += 1
                if patience and epochs_without_improvement >= patience:
                    print("Early stopping after {} epochs without improvement.".format(patience))
                    break

        if os.path.exists(model_filename):
            self.load_model(model_filename)

    def test(self, model_filename: str = DEFAULT_MODEL_PATH) -> float:
        # Always honor the requested checkpoint. After training, the in-memory
        # model contains the final epoch, which is not necessarily the best one.
        self.load_model_from_file(model_filename)
        test_loader = self.get_test_loader()
        _, acc, top5 = self._run_eval(test_loader, nn.CrossEntropyLoss())
        self.last_test_top5 = top5
        print("Test set accuracy: top-1={:.3f} top-5={:.3f}".format(acc, top5))
        return acc

    def idx_to_breed_name(self, breed_idx: int) -> str:
        if not self.class_names:
            train_dir = os.path.join(self.dataset_root, "train")
            self.class_names = ImageFolder(train_dir).classes
        return parse_breed_name(self.class_names[breed_idx])

    def predict(self, input_tensor: torch.Tensor) -> List[Prediction]:
        model = self._require_model()
        model.eval()
        with torch.inference_mode():
            logits = model(input_tensor.to(self.device))
            probs = F.softmax(logits, dim=1)
            k = min(3, probs.size(1))
            top_probs, top_idx = probs.topk(k, sorted=True)
        breed_probs = top_probs.squeeze(0).tolist()
        breed_idx = top_idx.squeeze(0).tolist()
        if not isinstance(breed_probs, list):
            breed_probs = [breed_probs]
            breed_idx = [breed_idx]
        names = [self.idx_to_breed_name(idx) for idx in breed_idx]
        return select_predictions(names, breed_probs, self.confidence_threshold)

    def predict_breed(self, image_file: str) -> List[Prediction]:
        with Image.open(image_file) as img:
            image = img.convert("RGB")
            image_tensor = self.eval_transforms(image).unsqueeze(0)
        return self.predict(image_tensor)


def _common_cli() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    common.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Checkpoint path")
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--epochs", type=int, default=50)
    common.add_argument("--lr", type=float, default=0.01)
    common.add_argument("--num-workers", type=int, default=None)
    common.add_argument("--patience", type=int, default=7, help="Early-stopping patience (0 disables)")
    common.add_argument("--device", default=None, help="cuda, mps, or cpu (default: auto)")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_cli()
    parser = argparse.ArgumentParser(
        prog="dog_breed_classifier.py",
        description="Train or run a ResNet-152 dog-breed classifier.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", parents=[common], help="Train on dogImages/ and save the best checkpoint")
    subparsers.add_parser("test", parents=[common], help="Evaluate a checkpoint on dogImages/test")

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Predict breed(s) for an image")
    predict_parser.add_argument("image", help="Path to a dog image")

    return parser


def _classifier_from_args(args: argparse.Namespace) -> DogBreedClassifier:
    device = torch.device(args.device) if args.device else None
    return DogBreedClassifier(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        amp=not args.no_amp,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Back-compat: `python dog_breed_classifier.py image.jpg`
    known_commands = {"train", "test", "predict"}
    if argv and not argv[0].startswith("-") and argv[0] not in known_commands:
        argv = ["predict"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command is None:
        parser.print_help()
        return 2

    classifier = _classifier_from_args(args)

    if command == "train":
        classifier.train(args.model, patience=args.patience)
        return 0
    if command == "test":
        classifier.test(args.model)
        return 0
    if command == "predict":
        if not os.path.exists(args.model):
            print("No checkpoint at {}. Train first or pass --model.".format(args.model), file=sys.stderr)
            return 1
        classifier.load_model_from_file(args.model)
        print(classifier.predict_breed(args.image))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
