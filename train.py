import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model to classify images.")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (vgg13 or resnet18)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    return parser.parse_args()

def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders

def build_model(arch, hidden_units):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch == "densenet":
        model = models.densenet169(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier_input_size = model.classifier.in_features
        classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    else:
        raise ValueError("Invalid architecture. Use 'vgg16' or 'densenet'")
        
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                dataset_size = len(dataloaders['train'].dataset)
            else:
                dataset_size = len(dataloaders['valid'].dataset)
            
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
    return model

def save_checkpoint(model, arch, image_datasets, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

def main():
    args = get_input_args()
    data_dir = args.data_dir
    image_datasets, dataloaders = load_data(data_dir)
    
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    
    save_checkpoint(model, args.arch, image_datasets, args.save_dir)

if __name__ == '__main__':
    main()
