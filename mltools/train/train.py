import torch
import torch.optim as optim


class Learner:

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, epochs):
        pass

    def test(self):
        pass
