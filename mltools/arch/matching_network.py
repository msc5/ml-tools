
import torch
import torch.nn as nn

from torchinfo import summary


class Conv(nn.Module):

    def __init__(self, fi, fo):
        super(Conv, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(fi, fo, 3, padding="same"),
            nn.BatchNorm2d(fo),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv_layer(x)


class Embed(nn.Module):

    def __init__(self, fi, fo):
        super(Embed, self).__init__()
        self.fo = fo
        self.embed = nn.Sequential(
            Conv(fi, fo),
            Conv(fo, fo),
            Conv(fo, fo),
            Conv(fo, fo),
        )

    def forward(self, x):
        _, _, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.embed(x)
        x = x.view(-1, self.fo)
        return x


class Classifier(nn.Module):

    def __init__(self, device):
        self.device = device
        super(Classifier, self).__init__()

    def forward(self, x, shapes):
        k, n, q, m = shapes
        y = torch.eye(k).repeat_interleave(n, dim=0).to(self.device)
        pred = torch.mm(x, y).log()
        return pred


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, s, t):
        n, q = s.shape[0], t.shape[0]
        dist = (
            t.unsqueeze(0).expand(n, q, -1) -
            s.unsqueeze(1).expand(n, q, -1)
        ).pow(2).sum(dim=2).T
        return dist


class MatchingNets(nn.Module):

    def __init__(self, device, fi):
        super(MatchingNets, self).__init__()
        self.device = device
        fo = 64
        self.f = Embed(fi, fo)
        self.g = Embed(fi, fo)
        self.distance = Distance()
        self.classify = Classifier(self.device)
        self.__name__ = 'MatchingNetwork'

    def forward(self, s, t):
        k, n, _, _, _ = s.shape
        q, m, _, _, _ = t.shape
        s = self.f(s)
        t = self.g(t)
        dist = -self.distance(s, t)
        attn = dist.softmax(dim=1)
        pred = self.classify(attn, (k, n, q, m))
        return pred


if __name__ == '__main__':
    k = 20
    n = 1
    m = 19
    c = 1
    s = 28
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MatchingNets(device, c).to(device)
    summary(model, input_size=[
        (k, n, c, s, s),
        (k, m, c, s, s)
    ], device=device)
