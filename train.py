import torch
# import torchvision
from torch import utils, nn, optim
from torchvision import datasets, transforms, models
import time, json
from get_input_args import get_input_args

def train(hyperparameters):
    [arch_models, epochs, rate_learning, units_hidden, gpu, dir_source, path_checkpoint, path_json] = hyperparameters
    
    if arch_models == 'vgg16':
        model = models.vgg16()
        nodes_input = model.classifier[0].in_features
        
    elif arch_models == 'alexnet':
        model = models.alexnet()
        nodes_input = model.classifier[1].in_features
        
    with open(path_json, 'r') as catf:
        cat_to_name = json.load(catf)
        
    nodes_output = len(cat_to_name)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
            nn.Linear(nodes_input, units_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.13),
            nn.Linear(units_hidden, 256),
            nn.ReLU(),
            nn.Dropout(p=0.17),
            nn.Linear(256, nodes_output),
            nn.LogSoftmax(dim=1)
            )
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = rate_learning)
    
    if gpu == 'gpu':
        print('Checking GPU available:')
        if torch.cuda.is_available():
            print('Using GPU')
            device = 'cuda'
        else:
            print('Using CPU instead')
            device = 'cpu'
            
        model.to(device)
        
        print('Traning Starts Here ====>')
        time_start = time.time()
        
        steps = 0
        every_print = 10
        losses_train, losses_valid = [], []
        for epoch in range(epochs):
            print('Epoch ===>', str(epoch))
            loss_train = 0
            for inputs, labels in dataloaders(dir_source, 'train'):
                inputs, labels = inputs.to(device), labels.to(device)
                steps += 1
                print('Step No.:', str(steps), end = '\r')
                
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss_train += loss.item()
                loss.backward()
                optimizer.step()
                
                if steps % every_print == 0:
                    loss_valid, accuracy_valid = 0, 0
                    model.eval()
                    
                    # Turn off gradients
                    with torch.no_grad():
                        for inputs, labels in dataloaders(dir_source, 'valid'):
                            inputs, labels = inputs.to(device), labels.to(device)
                            
                            logps = model.forward(inputs)
                            loss = criterion(logps, labels)
                            loss_valid += loss.item()
                            ps = torch.exp(logps)
                            p_top, class_top = ps.topk(1, dim = 1)
                            equals = class_top == labels.view(*class_top.shape)
                            accuracy_valid += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    # Model back to train mode
                    model.train()
                    
                    length_train = len(dataloaders(dir_source, 'train'))
                    length_valid = len(dataloaders(dir_source, 'valid'))
                    losses_train.append(loss_train / length_train)
                    losses_valid.append(loss_valid / length_valid)
                    
                    print(f'Epoch [{epoch + 1}:{epochs}] => ',
                          f'Train loss: ({loss_train/length_train:.3f}) '
                          f'Valid loss: ({loss_valid/length_valid:.3f}) '
                          f'Accuracy Valid: ({accuracy_valid/length_valid:.3f})')
    
    print('Training Completed ======/')
    print('Training Time', (time.time() - time_start), 'secs')
    
    checkpoint = {
        'epochs': epochs,
        'rate_learning': rate_learning,
        'classifier_model': classifier,
        'dict_state_model' : model.state_dict(),
        'dict_state_optimizer': optimizer.state_dict(),
        'class_to_idx': datasets_image(dir_source, 'train').class_to_idx
    }
    
    torch.save(checkpoint, path_checkpoint)
    print('Model saved, Location:', path_checkpoint)
    
def dir_data(dir_source, stage):
    return dir_source + '/' + stage

def transforms_data(stage):
    size_pic = 224
    means = [0.485, 0.456, 0.406]
    standard_deviations = [0.229, 0.224, 0.225]
    
    if stage == 'train':
        return transforms.Compose([
                transforms.RandomResizedCrop(size_pic, scale = (0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, standard_deviations)])
    
    elif stage == 'valid':
        return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size_pic),
                transforms.ToTensor(),
                transforms.Normalize(means, standard_deviations)])
    
    elif stage == 'test':
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(size_pic),
                transforms.ToTensor(),
                transforms.Normalize(means, standard_deviations)
            ])
    
def datasets_image(dir_source, stage):
    dataset_image = datasets.ImageFolder(dir_data(dir_source, stage), transform = transforms_data(stage))
    return dataset_image

def dataloaders(dir_source, stage):
    dataloader = utils.data.DataLoader(datasets_image(dir_source, stage), batch_size = 64, shuffle = True)
    return dataloader

def main():
    input_args = get_input_args()
    arch_models = input_args.arch
    epochs = input_args.epochs
    rate_learning = input_args.learning_rate
    units_hidden = input_args.hidden_units
    gpu = input_args.gpu
    dir_source = input_args.source_dir
    path_checkpoint = input_args.checkpoint_path
    path_json = input_args.json_path
    hyperparameters = [arch_models, epochs, rate_learning, units_hidden, gpu, dir_source, path_checkpoint, path_json]
    
    print('Inputs ====>')
    print('Pre trained model arch:', arch_models)
    print('Epochs:', epochs)
    print('Learning Rate:', rate_learning)
    print('Hidden Units:', units_hidden)
    print('GPU/CPU :', gpu)
    print('Directory Source :', dir_source)
    print('Path of Checkpoint:', path_checkpoint)
    print('Path of Json:', path_json)
    
    train(hyperparameters)
    
if __name__ == '__main__':
    main()

                          
        
    