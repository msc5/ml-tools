import torch
import torch.optim as optim

from rich import print
import os
import math

from ..arch.matching_network import MatchingNets
from ..arch.relation_network import RelationNetwork
from ..data.dataset import Dataset
from ..data.loaders import FewShotLoader


#  class Learner:

#      def __init__(self, model, dataset):
#          self.model = model
#          self.dataset = dataset
#          self.optimizer = optim.Adam(self.model.parameters())

#      def train(self, epochs):
#          pass

#      def test(self):
#          pass

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    k = 20
    n = 1
    m = 5

    model = MatchingNets(device, 1).to(device)
    #  model = RelationNetwork(1, 64, 64, k, n, m).to(device)

    #  path = 'datasets/omniglot/train'
    path = 'datasets/miniimagenet/train'

    dataset = Dataset(path)
    sampler = FewShotLoader(dataset, {'k': k, 'n': n, 'batch_size': m})

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    #  loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    for s, q, lab in sampler:

        pred = model(s, q)

        loss_t = loss_fn(pred, lab)
        loss = loss_t.item()

        correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
        acc = correct / pred.shape[0]

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        os.system('clear')
        #  def s(x): return 1 / (1 + math.exp(-x))
        print(f'Loss |{int(loss * 100) * "-":100}|\nAccuracy:{acc:>10.5f}')
