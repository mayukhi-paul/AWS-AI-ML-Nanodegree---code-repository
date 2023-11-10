import argparse
import torch
import json
from torchvision import models, transforms
from PIL import Image

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to image file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "densenet":
        model = models.densenet169(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def predict(image_path, model, topk, category_names, device):
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        class_to_idx = model.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        top_classes = [idx_to_class[i.item()] for i in top_class[0]]
        if category_names:
            with open(category_names, 'r') as f:
                cat_to_name = json.load(f)
            top_class_names = [cat_to_name[str(cls)] for cls in top_classes]
        else:
            top_class_names = top_classes
        return top_p[0].tolist(), top_classes, top_class_names

def main():
    in_args = get_input_args()
    image_path = in_args.input
    checkpoint_path = in_args.checkpoint
    top_k = in_args.top_k
    category_names = in_args.category_names
    device = torch.device("cuda" if in_args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(checkpoint_path)

    top_probabilities, top_classes, top_class_names = predict(image_path, model, top_k, category_names, device)

    print("Top", top_k, "classes and probabilities:")
    for i in range(top_k):
        print(f"{top_class_names[i]}: {top_probabilities[i] * 100:.2f}%")

if __name__ == "__main__":
    main()
