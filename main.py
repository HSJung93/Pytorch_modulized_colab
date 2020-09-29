if __name__ == "__main__":

    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from datasets import datasets
    from model import model_selection
    from training import train
    from efficientnet_pytorch import EfficientNet

    # !python3 main.py --dataset CIFAR10 --batch_size 100 --epochs 5 --model_name resnet18
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--dataset', default = 'CIFAR10', help = 'dataset')
    parser.add_argument('--batch_size', default = 50, type = int, help = 'batch size')
    parser.add_argument('--pretrained_weights', default = None, help = 'model path')
    parser.add_argument('--model_name', default = None, help = 'model')
    parser.add_argument('--efficientnet', default = 'b0', help = 'EfficientNet architecture')
    parser.add_argument('--classes', default = 10, help = 'number of classes')
    parser.add_argument('--epochs', default = 10, type = int, help = 'epochs')
    parser.add_argument('--learning_rate', default = 1e-3, type = int, help = 'learning rate') #?
    parser.add_argument('--interval', default = 1, type = int, help = 'model.save')

    args = parser.parse_args()
    print(args)

    trainloader, testloader = datasets(args.dataset, args.batch_size)
    print("Completed reading the dataset")

    if args.model_name is None:
        model = EfficientNet.from_pretrained('efficientnet-' + args.efficientnet, num_classes = args.classes) # for 3 channel input
    else:
        model = model_selection(args.model_name, args.classes)
    
    if args.pretrained_weights is not None:
        model.load_state_dict(torch.load(args.pretrained_weights))

    print("Completed loading the model")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    train(trainloader, testloader, model, criterion, optimizer, args.epochs, args.interval)
    print("Completed training the model")

    




