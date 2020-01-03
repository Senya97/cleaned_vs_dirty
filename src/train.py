import argparse
from os.path import join
from tqdm import tqdm

import torch
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from constants import DATASET_PATH, BATCH_SIZE
from constants import NUM_EPOCH, SCHEDULER, LOSS, OPTIMIZER, MODEL
from constants import TRANSFORM


def create_argument_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='')
    return parser


def main(args):
    DATASET_PATH = args.dataset_path
    TRAIN_PATH = join(DATASET_PATH, 'train/')
    VAL_PATH = join(DATASET_PATH, 'val/')

    # loading dataset
    train_dataset = ImageFolder(TRAIN_PATH, TRANSFORM)
    val_dataset = ImageFolder(VAL_PATH, TRANSFORM)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=1,
                                  )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=1,
                                )

    # model initialization
    model = MODEL
    for param in model.parameters():
        param.requires_grad = False
    model.fc = Linear(model.fc.in_features, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train model
    for epoch in tqdm(range(NUM_EPOCH)):
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                SCHEDULER.step()
                model.train()
            else:
                dataloader = val_dataloader
                model.eval()
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                OPTIMIZER.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = LOSS(preds, labels)
                    if phase == 'train':
                        loss_value.backward()
                        OPTIMIZER.step()
    return


if __name__ == '__main__':
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    main(arguments)
