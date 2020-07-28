#!/usr/bin/env python3
# -*- coding: utf-8 -*
#   Example call:
#   python train.py data_dir
#   python train.py data_dir --save_dir out --arch "vgg13" --epochs 10 --gpu
##

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from time import monotonic
from parse_args import parse_train_args

MEAN_NORM = [0.485, 0.456, 0.406]
STD_NORM = [0.229, 0.224, 0.225]

def get_test_transforms():
    return transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize(MEAN_NORM, STD_NORM)])

# TODO: deduplication
def get_train_transforms():
    return transforms.Compose([transforms.RandomRotation(30),
                               transforms.Resize(256),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(MEAN_NORM, STD_NORM)])

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
    
    model.classifier = nn.Sequential(nn.Linear(args.input_size, args.hidden_1_size),
                               nn.ReLU(),
                               nn.Dropout(p=args.dropout_rate),
                               nn.Linear(args.hidden_1_size, args.hidden_2_size),
                               nn.ReLU(),
                               nn.Dropout(p=args.dropout_rate),
                               nn.Linear(args.hidden_2_size, args.output_size),
                               nn.LogSoftmax(dim=1))
    return model

def train_model(model, optimizer, criterion, device, args):
    # Load training and validation data
    train_data = datasets.ImageFolder(args.data_dir + '/train', transform=get_train_transforms())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_data = datasets.ImageFolder(args.data_dir + '/valid', transform=get_test_transforms())
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    model.class_to_idx = train_data.class_to_idx
    model.to(device)
    
    steps = 1
    running_loss = 0
    print_every = 5
    for epoch in range(args.epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                print("Epoch: {}/{}..".format(epoch+1, args.epochs),
                      "Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
                      "Validation Loss: {:.3f}..".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()

def test_model(model, criterion, device, args):
    model.eval()
    test_data = datasets.ImageFolder(args.data_dir + '/test', transform=get_test_transforms())
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        print("Test Loss: {:.3f}..".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(test_accuracy/len(testloader)))
    
def save_checkpoint(model, optimizer, args):
    checkpoint = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'input_size': args.input_size,
        'hidden_1_size': args.hidden_1_size,
        'hidden_2_size': args.hidden_2_size,
        'dropout': args.dropout_rate,
        'output_size': args.output_size,
        'class_to_idx': model.class_to_idx,
        'classifier_state': model.classifier.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, args.save_dir + '/checkpoint.' + args.arch)

def main():
    start_time = monotonic()
    args = parse_train_args()
    
    model = get_model(args)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()
    # Enable GPU only if available and configured
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    
    train_model(model, optimizer, criterion, device, args)
    test_model(model, criterion, device, args)
    save_checkpoint(model, optimizer, args)
    
    end_time = monotonic()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))

if __name__ == "__main__":
    main()