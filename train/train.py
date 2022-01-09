import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

import os
import time
import shutil

import arch
import data

from .logger import Logger


def train(
        name,
        model,
        dataloader,
        callback,
        optimizer,
        scheduler,
        loss_fn,
        epochs,
        device,
):

    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    print(device)
    print(epochs, num_batches, batch_size)

    # Setup files
    model_type = model.__name__
    type_dir = os.path.join('saves', model_type)
    model_dir = os.path.join(type_dir, name)
    model_path = os.path.join(model_dir, 'model')
    log_path = os.path.join(model_dir, 'log')
    if not os.path.exists('saves'):
        os.makedirs('saves')
    if not os.path.exists(type_dir):
        os.makedirs(type_dir)
    if os.path.exists(model_dir):
        ans = input(
            f'{model_type + ": " + name} has already been trained. Overwrite save files? (y/n)\n',
        )
        if ans == 'y' or ans == 'Y':
            pass
        else:
            return
    else:
        os.makedirs(model_dir)
    config_path = os.path.join(model_dir, 'config.json')
    shutil.copyfile('config.json', config_path)

    # Initialize Logger
    logger = Logger(epochs, num_batches, log_path)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.zero_grad()

    # Print header
    print(logger.header())
    tic = time.perf_counter()

    for i in range(epochs):

        t = tqdm(
            dataloader,
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, (train_ds, test_ds) in enumerate(t):
            train_results = callback(
                model,
                train_ds,
                optimizer,
                loss_fn,
                device,
                train=True
            )
            with torch.no_grad():
                test_results = callback(
                    model,
                    test_ds,
                    None,
                    loss_fn,
                    device,
                    train=False
                )
            toc = time.perf_counter()
            log = logger.log((*train_results, *test_results), toc - tic)
            t.set_description(log)

        print(log)
        torch.save(model.state_dict(), model_path)
        scheduler.step()


def omniglotCallBack(
        model,
        inputs,
        optimizer,
        loss_fn,
        device,
        train=True
):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    (s, t), classes = inputs

    pred = model(s, t)

    k = int(pred.shape[1])
    m = int(pred.shape[0] / k)

    lab = torch.eye(k).repeat_interleave(m, dim=0).to(device)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    if train:
        optimizer.zero_grad()
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss, acc


def build_model(model_config, data_config):
    channels = data_config['channels']
    if model_config['arch'] == 'CustomNetwork':
        meta_layers = data_config['meta_layers']
        model = arch.CustomNetwork(meta_layers, channels, 64, 16, device)
        return model


if __name__ == '__main__':

    main_config = json.load(open('config.json'))
    dataset_config = json.load(open(os.path.join('datasets', 'datasets.json')))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = main_config['models']
    experiments = main_config['experiments']
    todo = main_config['todo']

    for item in todo:
        model_name, exp_name = item
        model_config = models[model_name]
        exp_config = experiments[exp_name]
        dataset_name = exp_config['dataset']
        data_config = dataset_config[dataset_name]
        model = build_model(model_config, data_config)
        loader = data.fewshot.FewShotDataset(
            directory='datasets/' + dataset_name,
            device=device
        )
        if exp_config['todo'] == 'train':
            #  train(
            #      model_name,
            #      model,
            #      loader,
            #      optimizer,
            #      scheduler,

            #  )
        elif exp_config['todo'] == 'test':
