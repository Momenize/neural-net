import numpy as np
import torch
from imageio import imread
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split

# train_img = []
# for img_name in tqdm(train['Image_Name']):
#     # defining the image path
#     image_path = os.path.join('images/autoencoder/PetImages', str(img_name) + '.JPG')
    
#     # reading the image
#     img = imread(image_path, as_gray=True )
    
                      
#     # normalizing the pixel values
#     img /= 255.0
#     # converting the type of pixel to float 32
#     img = img.astype('float32')
#     # appending the image into the list
#     train_img.append(img)
# image_paths = glob.glob('images/autoencoder/*.JPG')
image_paths = glob.glob('images/autoencoder/PetImages')
train_img = []
for image_path in tqdm(image_paths):
    # reading the image
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    train_img.append(img)
# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_img, test_size = 0.2)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

# converting training images into torch format
train_x = train_x.reshape(81, 1, 28, 28)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# Convert validation data to torch tensors (if not already)
val_x = val_x.reshape(val_x.shape[0], 1, 28, 28)
val_x = torch.from_numpy(val_x)
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

# Create TensorDatasets and DataLoaders
batch_size = 16
train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Writing our model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate model, loss, optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        optimizer.zero_grad()
        output = model(data.float())
        loss = criterion(output, data.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            output = model(data.float())
            loss = criterion(output, data.float())
            val_loss += loss.item() * data.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")