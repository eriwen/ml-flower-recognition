#!/usr/bin/env python3
# -*- coding: utf-8 -*
#   Example call:
#   python predict.py path/to/image.jpg path/to/model.chk
#   python predict.py image.jpg model.chk --top_k 3 --category_names cat_name.json --gpu
##

import json
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
import torchvision.transforms.functional as TF
from parse_args import parse_predict_args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
        returns an Numpy array normalized to ImageNet standards.
    '''
    image = Image.open(image)
    image = TF.resize(image, 256)
    image = TF.center_crop(image, 224)
    tensor = TF.to_tensor(np.asarray(image))
    return TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_model(args):
    if args.arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif args.arch == "densenet":
        model = models.densenet121(pretrained=True)
    elif args.arch == "resnet":
        model = models.resnet18(pretrained=True)
    elif args.arch == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint_path)
    model = get_model(args)
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_1_size']),
                           nn.ReLU(),
                           nn.Dropout(p=checkpoint['dropout']),
                           nn.Linear(checkpoint['hidden_1_size'], checkpoint['hidden_2_size']),
                           nn.ReLU(),
                           nn.Dropout(p=checkpoint['dropout']),
                           nn.Linear(checkpoint['hidden_2_size'], checkpoint['output_size']),
                           nn.LogSoftmax(dim=1))
    classifier.load_state_dict(checkpoint['classifier_state'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return model, optimizer

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    model.eval()
    image = process_image(image_path)
    # invoke model to get ln(probabilities)
    with torch.no_grad():
        logps = model.forward(image.unsqueeze_(0))
    ps = torch.exp(logps)
    return ps.topk(topk, dim=1)

def main():
    args = parse_predict_args()
    
    #image_path = './flowers/test/1/image_06743.jpg'
    image_path = args.image_path
    model, optimizer = load_checkpoint(args)
    probabilities, categories = predict(image_path, model, args.top_k)
    numpy_probabilities = probabilities.detach().cpu().numpy()[0]
    numpy_categories = categories.detach().cpu().numpy()[0]

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    idx_to_category = {v: k for k, v in model.class_to_idx.items()}
    predicted_flowers = [cat_to_name[idx_to_category[cat]] for cat in numpy_categories]
    
    print("{:>20} | Probability".format('Flower Species'))
    print("-" * 34)
    for i in range(len(predicted_flowers)):
        print("{:>20} | {:.3f}".format(predicted_flowers[i].title(), numpy_probabilities[i]))

if __name__ == "__main__":
    main()