import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
from PIL import Image

import dog_breed_classifier as mod
from dog_breed_classifier import (
    DogBreedClassifier,
    build_parser,
    main,
    parse_breed_name,
    select_predictions,
)


def _cpu_classifier(**kwargs):
    defaults = dict(device=torch.device("cpu"), num_workers=0)
    defaults.update(kwargs)
    return DogBreedClassifier(**defaults)


def _write_jpeg(path, color=(30, 90, 180)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path, format="JPEG")


class TinyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class ParseBreedNameTests(unittest.TestCase):
    def test_udacity_folder_name(self):
        self.assertEqual(parse_breed_name("001.Affenpinscher"), "Affenpinscher")

    def test_underscores_become_spaces(self):
        self.assertEqual(parse_breed_name("124.Poodle_standard"), "Poodle standard")

    def test_name_without_prefix(self):
        self.assertEqual(parse_breed_name("Beagle"), "Beagle")


class SelectPredictionsTests(unittest.TestCase):
    def test_confident_top1(self):
        names = ["Beagle", "Basset Hound", "Harrier"]
        probs = [0.91, 0.05, 0.02]
        self.assertEqual(select_predictions(names, probs), [("Beagle", 0.91)])

    def test_uncertain_top3(self):
        names = ["Beagle", "Basset Hound", "Harrier"]
        probs = [0.42, 0.31, 0.11]
        self.assertEqual(
            select_predictions(names, probs),
            [("Beagle", 0.42), ("Basset Hound", 0.31), ("Harrier", 0.11)],
        )

    def test_empty(self):
        self.assertEqual(select_predictions([], []), [])


class TransformTests(unittest.TestCase):
    def test_eval_transform_shape(self):
        classifier = _cpu_classifier()
        image = Image.new("RGB", (640, 480), color=(12, 64, 128))
        tensor = classifier.eval_transforms(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_train_transform_shape(self):
        classifier = _cpu_classifier()
        image = Image.new("RGB", (640, 480), color=(12, 64, 128))
        tensor = classifier.train_transforms(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))


class LoaderTests(unittest.TestCase):
    def test_get_loader_reads_imagefolder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_jpeg(os.path.join(tmpdir, "016.Beagle", "a.jpg"))
            classifier = _cpu_classifier(batch_size=1)
            loader = classifier.get_loader(tmpdir, train=True)
            self.assertEqual(classifier.class_names, ["016.Beagle"])
            images, labels = next(iter(loader))
            self.assertEqual(tuple(images.shape), (1, 3, 224, 224))
            self.assertEqual(tuple(labels.shape), (1,))


class TrainLoopTests(unittest.TestCase):
    def _write_split(self, root, split, class_name, count):
        for i in range(count):
            _write_jpeg(
                os.path.join(root, split, class_name, "{}.jpg".format(i)),
                color=(i * 10, 40, 200),
            )

    def test_train_runs_one_epoch_and_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_split(tmpdir, "train", "001.Beagle", 2)
            self._write_split(tmpdir, "valid", "001.Beagle", 1)
            self._write_split(tmpdir, "test", "001.Beagle", 1)
            self._write_split(tmpdir, "train", "002.Basset_hound", 2)
            self._write_split(tmpdir, "valid", "002.Basset_hound", 1)
            self._write_split(tmpdir, "test", "002.Basset_hound", 1)
            checkpoint = os.path.join(tmpdir, "best_model.pt")
            classifier = _cpu_classifier(
                dataset_root=tmpdir,
                batch_size=2,
                num_epochs=1,
                lr=0.01,
            )
            with mock.patch.object(mod, "build_resnet152", side_effect=lambda n: TinyNet(n)):
                classifier.train(checkpoint, patience=0)
                acc = classifier.test(checkpoint)
                preds = classifier.predict_breed(
                    os.path.join(tmpdir, "test", "001.Beagle", "0.jpg")
                )
            self.assertTrue(os.path.exists(checkpoint))
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
            self.assertTrue(preds)
            self.assertIn(preds[0][0], ("Beagle", "Basset hound"))
            self.assertGreaterEqual(preds[0][1], 0.0)
            self.assertEqual(
                [parse_breed_name(name) for name in classifier.class_names],
                ["Beagle", "Basset hound"],
            )


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        classifier = _cpu_classifier(num_breeds=3)
        classifier.class_names = ["001.Affenpinscher", "002.Afghan_hound", "003.Airedale_terrier"]
        classifier.model = nn.Linear(4, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            classifier.save_model(path)

            loaded = _cpu_classifier(num_breeds=3)
            loaded.model = nn.Linear(4, 3)
            loaded.load_model(path)
            self.assertEqual(loaded.class_names, classifier.class_names)
            for left, right in zip(classifier.model.parameters(), loaded.model.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_load_strips_dataparallel_prefix(self):
        classifier = _cpu_classifier(num_breeds=2)
        classifier.model = nn.Linear(3, 2)
        state = {"module." + key: value for key, value in classifier.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pt")
            torch.save(state, path)
            loaded = _cpu_classifier(num_breeds=2)
            loaded.model = nn.Linear(3, 2)
            loaded.load_model(path)
            for left, right in zip(classifier.model.parameters(), loaded.model.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_idx_to_breed_name_uses_cached_classes(self):
        classifier = _cpu_classifier()
        classifier.class_names = ["016.Beagle"]
        self.assertEqual(classifier.idx_to_breed_name(0), "Beagle")

    def test_load_model_from_file_uses_checkpoint_num_breeds(self):
        classifier = _cpu_classifier(num_breeds=2)
        classifier.class_names = ["001.Beagle", "002.Basset_hound"]
        classifier.model = TinyNet(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            classifier.save_model(path)
            loaded = _cpu_classifier(num_breeds=133)
            with mock.patch.object(mod, "build_resnet152", side_effect=lambda n: TinyNet(n)):
                loaded.load_model_from_file(path)
            self.assertEqual(loaded.num_breeds, 2)
            self.assertEqual(loaded.model.fc.out_features, 2)
            self.assertEqual(loaded.class_names, classifier.class_names)


class CliTests(unittest.TestCase):
    def test_parser_subcommands(self):
        parser = build_parser()
        train_args = parser.parse_args(["train", "--epochs", "2"])
        self.assertEqual(train_args.command, "train")
        self.assertEqual(train_args.epochs, 2)

        predict_args = parser.parse_args(["predict", "dog.jpg", "--model", "weights.pt"])
        self.assertEqual(predict_args.command, "predict")
        self.assertEqual(predict_args.image, "dog.jpg")
        self.assertEqual(predict_args.model, "weights.pt")

    def test_help_exit_code_without_command(self):
        self.assertEqual(main([]), 2)

    def test_legacy_image_path_is_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "missing.pt")
            code = main(["a_dog.jpg", "--model", missing])
            self.assertEqual(code, 1)

    def test_predict_without_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "missing.pt")
            code = main(["predict", "unused.jpg", "--model", missing])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
