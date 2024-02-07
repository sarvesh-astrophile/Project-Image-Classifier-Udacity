import json
from get_input_args import get_input_args
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchvision import models
import numpy as np
from PIL import Image

def image_process(image):
    width, height = image.size
    ratio_aspect = width / height
    
    if width < height:
        width_new = 256
        height_new = int(width_new / ratio_aspect)
    else:
        new_height = 256
        new_width = int(new_heigth * aspect_ratio)
        
    image_resized = image.resize((width_new, height_new))
    
    left = (width_new - 224) / 2
    top = (height_new - 224) / 2
    right = (width_new + 224) / 2
    bottom = (height_new + 224) / 2
    image_cropped = image_resized.crop((left, top, right, bottom))
    
    image_np = np.array(image_cropped)
    
    image_np = image_np / 255.0
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - means) / stds
    
    image_output = image_np.transpose((2, 0, 1))
    
    return image_output

def predict(path_image, path_checkpoint, arch_models, gpu, topk = 5):
    model, class_to_idx = checkpoint_load(path_checkpoint, arch_models)
    
    if gpu == 'gpu':
        print('Checking GPU availablity:')
        if torch.cuda.is_available():
            print('Using GPU')
            device = 'cuda'
        else:
            print('Using CPU instead')
            device = 'cpu'
        model.to(device)
        
        with Image.open(path_image) as image:
            data_image_process = image_process(image)
        img = torch.from_numpy(data_image_process)
        img = img.to(device)
        reshape_img = img.unsqueeze(0).float()
        
        with torch.no_grad():
            model.eval()
            model.to(device)
            
            logps = model(reshape_img)
            ps = torch.exp(logps)
            
            probs, labels = ps.topk(topk, dim = 1)
            probs_rounded = [round(num, 4) for num in probs.tolist()[0]]
            class_to_idx_convert = {class_to_idx[i]: i for i in class_to_idx}
            classes = []
            for label in labels.cpu().numpy()[0]:
                classes.append(class_to_idx_convert[label])
                
            return probs_rounded, classes

def checkpoint_load(path_checkpoint, arch_models):
    checkpoint = torch.load(path_checkpoint)
    
    if arch_models == 'vgg16':
        model = models.vgg16()
    elif arch_models == 'alexnet':
        model = models.alexnet()
    
    model.classifier = checkpoint['classifier_model']
    model.load_state_dict = checkpoint['dict_state_model']
    model.optimizer = checkpoint['dict_state_optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, model.class_to_idx

def imshow(image, ax = None, title = None):
    if ax is None:
        fig, ax = plt.subplots()
        
    image = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax

def main():
    args_input = get_input_args()
    arch_models = args_input.arch
    gpu = args_input.gpu
    path_checkpoint = args_input.checkpoint_path
    path_img = args_input.image_path
    
    print("Inputs ===>")
    print('Arch:', arch_models)
    print('using GPU/CPU', gpu)
    print('Checkpoint Path:', gpu)
    print('Image Path:', path_img)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    probs_predict, classes_predict = predict(path_img, path_checkpoint, arch_models, gpu)
    
    print('The Prediction Results ==>')
    print('File selected: ', path_img)
    print('Top 5 predicted flower classes:', classes_predict)
    print('and their prebablities are', probs_predict)
    
    class_image_correct = path_img.split("/")[2]
    
    classes = []
    for class_predict in classes_predict:
        classes.append(cat_to_name[class_predict])


if __name__ == '__main__':
    main()
    
