'''
this file holds the model and the transforms used in the model training
'''


import torch
import torch.nn as nn
import torchvision.models as models

class MultiOutputEfficientNetV2(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(MultiOutputEfficientNetV2, self).__init__()
        
        # Load a pre-trained EfficientNetV2 model
        self.efficientnet = models.efficientnet_v2_s(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale (1 channel) images
        self.efficientnet.features[0][0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Remove the final fully connected layer
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # We’ll add separate heads

        # Task-specific output layers
        self.task1_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task1)
        )
        
        self.task2_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task2)
        )
        
    def forward(self, x):
        # Forward through the shared EfficientNetV2 layers
        x = self.efficientnet(x)
        
        # Separate outputs for each task
        output_task1 = self.task1_head(x)
        output_task2 = self.task2_head(x)
        
        return output_task1, output_task2


class MultiOutputResNet(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(MultiOutputResNet, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept grayscale (1 channel) images
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # We’ll add separate heads
        
        # Task-specific output layers
        self.task1_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task1)
        )
        
        self.task2_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task2)
        )
        
    def forward(self, x):
        # Forward through the shared ResNet layers
        x = self.resnet(x)
        
        # Separate outputs for each task
        output_task1 = self.task1_head(x)
        output_task2 = self.task2_head(x)
        
        return output_task1, output_task2


def get_model(model_name:str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if(model_name == "effnetv2"):
        model = MultiOutputEfficientNetV2(num_classes_task1=4, num_classes_task2=4)
        model_weight_path = r"./classification_model/effnetv2_s_best_val_model.pth"

    elif(model_name == "resnet18"):
        model = MultiOutputResNet(num_classes_task1=4, num_classes_task2=4)
        model_weight_path = r"./classification_model/resnet18_best_val_model.pth"
    else:
        raise ValueError("Invalid model name specified. Couldn't load model")
    
    # have to use abs path here, otherwise error

    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device(device))['state_dict'])
    
    return model


# transform
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((124, 124)),
    # transforms.RandomApply([transforms.RandomResizedCrop(124, scale=(0.9, 1.0))], p=0.5),  # Randomly apply crop & resize
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.7),  # Perspective distortion
    # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.5), # Random ColorJitter
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),  # Random Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization
])


import torch

# Function to make predictions using the model
def predict(img, device='cpu'):
    '''
    first output = component class
    sedond output = component orientation
    '''
    img = img.convert("L") # converting img to grayscale
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model('effnetv2')
    model.to(device)  # Move model to device (GPU or CPU)
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    # Move the input data to the same device as the model
    input_data = transform(img)

    with torch.no_grad():  # No need to track gradients during inference
        # Get the predictions from the model (two outputs for multi-task)
        outputs = model(input_data.unsqueeze(0))
        
        # Assuming model returns two outputs for two tasks
        preds_task1, preds_task2 = outputs
        
        # Get the class with the highest probability for each task
        _, preds_task1 = torch.max(preds_task1, 1)
        _, preds_task2 = torch.max(preds_task2, 1)

    return preds_task1.cpu().numpy().item(), preds_task2.cpu().numpy().item()

