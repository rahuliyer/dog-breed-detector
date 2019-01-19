import torch

from torch.utils.data import DataLoader

from torchvision.models import resnet152
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.optim import SGD

import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import accuracy_score

import os

import sys

class DogBreedClassifier:
    def __init__(self):
        super().__init__()

        self.model = None
        self.num_breeds = 133
        self.lr = 0.01
        self.num_epochs = 50
        self.batch_size = 256
        self.image_size = 224

        self.dataset_root = 'dogImages'

        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = None

    def get_loader(self, datadir):
        ds = ImageFolder(datadir, self.transforms)

        return DataLoader(ds, self.batch_size, shuffle=True)

    def get_train_loader(self):
        return self.get_loader(self.dataset_root + '/train')

    def get_valid_loader(self):
        return self.get_loader(self.dataset_root + '/valid')

    def get_test_loader(self):
        return self.get_loader(self.dataset_root + '/test')

    def freeze_weights(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def get_model(self):
        self.model = resnet152(pretrained=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_breeds)

        # Freeze the weights of everything but the last layer
        layers = [layer for layer in self.model.children()]
        for i in range(len(layers) - 1):
            self.freeze_weights(layers[i])

        self.model = nn.DataParallel(self.model)
        self.model.cuda()

    def save_model(self, filename='best_model.pt'):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename='best_model.pt'):
        self.model.load_state_dict(torch.load(filename))

    def train(self):
        self.get_model()

        self.model.train()

        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = self.get_train_loader()
        valid_loader = self.get_valid_loader()

        print_every = 1

        best_valid_loss = np.Inf
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            for batch_nr, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()

                preds = self.model(inputs)

                loss = loss_fn(preds, targets)
                loss.backward()

                optimizer.step()

                train_loss = train_loss + ((1 / (batch_nr + 1)) * (loss.data - train_loss))

                if batch_nr % print_every == 0:
                    valid_loss = 0.0
                    self.model.eval()
                    v_preds = []
                    v_labels = []
                    for v_batch_nr, (inputs, targets) in enumerate(valid_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()

                        preds = self.model(inputs)
                        loss = loss_fn(preds, targets)

                        v_preds.extend(torch.argmax(preds, dim=1).tolist())
                        v_labels.extend(targets.tolist())

                        valid_loss = valid_loss + ((1 / (v_batch_nr + 1)) * (loss.data - valid_loss))

                    print(
                        "Epoch #{}, batch #{}: train loss: {}, valid loss:{}, validation accuracy: {}".format(
                            epoch,
                            batch_nr,
                            train_loss,
                            valid_loss,
                            accuracy_score(v_labels, v_preds)
                        ))

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss

                        print("Lowest validation loss; saving model...")
                        self.save_model()

                    self.model.train()

    def test(self):
        self.get_model()
        self.load_model()

        self.model.eval()

        test_loader = self.get_test_loader()

        preds = []
        labels = []
        for inputs, targets in test_loader:

            res = F.softmax(self.model(inputs), dim=1)

            preds.extend(torch.argmax(res, dim=1).tolist())
            labels.extend(targets.tolist())

        print("Test set accuracy: {}".format(accuracy_score(labels, preds)))

    def idx_to_breed_name(self, breed_idx):
        self.get_train_loader()
        classes = ImageFolder(self.dataset_root + '/train').classes

        return classes[breed_idx].split('.')[1]

    def predict(self, input):
        self.model.eval()

        input = input.cuda()

        preds = F.softmax(self.model(input), dim=1)
        breed_probs, breed_idx = preds.topk(3, sorted=True)

        breed_probs = breed_probs.squeeze().tolist()
        breed_idx = breed_idx.squeeze().tolist()

        if breed_probs[0] >= 0.8:
            return [(self.idx_to_breed_name(breed_idx[0]), breed_probs[0])]
        else:
            return [
                (self.idx_to_breed_name(breed_idx[0]), breed_probs[0]),
                (self.idx_to_breed_name(breed_idx[1]), breed_probs[1]),
                (self.idx_to_breed_name(breed_idx[2]), breed_probs[2])
            ]

    def predict_breed(self, image_file):
        img = Image.open(image_file).convert('RGB')

        image_tensor = self.transforms(img).unsqueeze(0)

        return self.predict(image_tensor)

    def load_model_from_file(self, filename):
        self.get_model()
        self.load_model(filename)

if __name__ == '__main__':
    model_filename = 'best_model.pt'
    classifier = DogBreedClassifier()
    if os.path.exists(model_filename):
        print("Pre-trained model found. Loading...")
        classifier.load_model_from_file(model_filename)
    else:
        classifier.train()
        classifier.test()

    print(classifier.predict_breed(sys.argv[1]))
