#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description='Argparser for train.py')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory within which to store results')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Neural Network Learning Rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--hidden_1_size', type=int, default=2048, help='Neurons in hidden layer 1')
    parser.add_argument('--hidden_2_size', type=int, default=512, help='Neurons in hidden layer 2')
    parser.add_argument('--output_size', type=int, default=102, help='Number of categories for output')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate between layers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU for training')
    parser.add_argument('--arch', type=str, default='vgg', help='Pretrained ImageNet Model ID')
    parser.add_argument('data_dir', type=str, nargs='?', help='Path to data directory')
    return parser.parse_args()

def parse_predict_args():
    parser = argparse.ArgumentParser(description='Argparser for predict.py')
    parser.add_argument('--arch', type=str, default='vgg', help='Pretrained ImageNet Model ID')
    parser.add_argument('--top_k', type=int, default=5, help='Number of categories to display')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU for inference')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='JSON Map from Flower ID to Name')
    parser.add_argument('image_path', type=str, nargs='?', help='Path to image file')
    parser.add_argument('checkpoint_path', type=str, nargs='?', help='Path to model checkpoint')
    return parser.parse_args()
