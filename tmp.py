"""
Transformaly Training Script
"""
import argparse
import logging
import pickle
import os

import torch.nn
from utils import print_and_add_to_log, get_datasets_for_ViT, \
    Identity, freeze_finetuned_model, train, plot_graphs, \
    extract_fetures
from os.path import join
from pytorch_pretrained_vit.model import AnomalyViT, ViT

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset',default='cifar10')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--lr', default=0.0001,
                        help='Learning rate value')
    parser.add_argument('--eval_every', type=int, default=2,
                        help='Will evaluate the model ever <eval_every> epochs')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--plot_every_layer_summarization', default=False, action='store_true',
                        help='plot the per layer AUROC')
    parser_args = parser.parse_args()
    args = vars(parser_args)


    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = True
    BASE_PATH = 'experiments'

    if args['dataset'] == 'cifar10':
        _classes = range(10)
    elif args['dataset'] == 'fmnist':
        _classes = range(10)
    elif args['dataset'] == 'cifar100':
        _classes = range(20)
    elif args['dataset'] == 'cats_vs_dogs':
        _classes = range(2)
    elif args['dataset'] == 'dior':
        _classes = range(19)
    else:
        raise ValueError(f"Does not support the {args['dataset']} dataset")

    if args['use_imagenet']:
        MODEL_NAME = 'B_16_imagenet1k'
    else:
        MODEL_NAME = 'B_16'

    for c in range(10):
        for filter in range(3):
            model = ViT(MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model.eval()

            extract_fetures(base_path=BASE_PATH,
                            data_path=args['data_path'],
                            datasets=[args['dataset']],
                            model=model,
                            logging=logging,
                            calculate_features=True,
                            unimodal_vals=[args['unimodal']],
                            manual_class_num_range=[c],
                            output_train_features=True,
                            output_test_features=True,
                            use_imagenet=args['use_imagenet'],
                            filter=filter)
