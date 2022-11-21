import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from IPython.display import clear_output
from skimage.io import imread
from skimage.transform import resize
import os
import re
import random
import time
from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
# %matplotlib nbagg
# %matplotlib inline
import seaborn as sns
import pandas as pd
import math 
import torch
from torch import nn, Tensor, sigmoid
from torch.nn.functional import softplus
from torch.distributions import Distribution, Bernoulli, Normal
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from collections import defaultdict
sns.set_style("whitegrid")

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from sklearn import metrics
import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, Bernoulli

sns.set_style("whitegrid")

def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

def compute_confusion_matrix(target, pred, normalize=None):
    return metrics.confusion_matrix(
        target.detach().cpu().numpy(), 
        pred.detach().cpu().numpy(),
        normalize=normalize
    )

start_time = time.time()
metadata = pd.read_csv('/Users/mikkelrasmussen/mnt/deep_learning_project/data/metadata.csv', engine="pyarrow")
print("pd.read_csv wiht pyarrow took %s seconds" % (time.time() - start_time))

DMSO_indx = metadata.index[metadata['moa'] == 'DMSO']
DMSO_drop_indices = np.random.choice(DMSO_indx, size=260360, replace=False)

metadata_subsampled = metadata.drop(DMSO_drop_indices).reset_index()

metadata_subsampled.groupby("moa").size().reset_index(name='counts').sort_values(by="counts", ascending=False)

# Map from class name to class index
classes = {index: name for name, index in enumerate(metadata["moa"].unique())}
classes_inv = {v: k for k, v in classes.items()}
classes
result_dir = 'results/'

class SingleCellDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, images_folder, class_map, 
                 mode='train', transform = None):
        self.df = annotation_file
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = class_map
            

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        filename = self.df.loc[index, "Single_Cell_Image_Name"]
        label = self.class2index[self.df.loc[index, "moa"]]
        subfolder = re.search("(.*)_", filename).group(1)
        image = np.load(os.path.join(self.images_folder, subfolder, filename))
        if self.transform is not None:
            image = self.transform(image.astype(np.float32))
        return image, label

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, Bernoulli


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        z = self.mu + self.sigma * self.sample_epsilon()
        return z
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        z = self.mu + self.sigma * self.sample_epsilon()
        return z
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        #var = self.sigma ** 2
        #log_scale = self.sigma.log()
        #return -((z - self.mu) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        
        return -((z-self.mu)**2) / (2*self.sigma**2) - self.sigma.log() - math.log(math.sqrt(2 * math.pi))
        
        #return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)

class ReparameterizedDiagonalGaussianWithSigmoid(ReparameterizedDiagonalGaussian):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        super().__init__(mu, log_sigma)
    
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        z = self.mu + self.sigma * self.sample_epsilon()
        return sigmoid(z)

batch_size = 64

# The loaders perform the actual work
images_folder = "/Users/mikkelrasmussen/mnt/deep_learning_project/data/singh_cp_pipeline_singlecell_images"
train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.flatten(x)),
     transforms.Lambda(lambda x: x/x.max())]
)

train_set = SingleCellDataset(images_folder=images_folder, 
                              annotation_file=metadata_subsampled, 
                              transform=train_transforms,
                              class_map=classes)

# Define the size of the train, validation and test datasets
data_prct = 0.01
train_prct = 0.8

data_amount = int(len(metadata_subsampled) * data_prct)
train_size = int(train_prct * data_amount)
val_size = (data_amount - train_size) // 2
test_size = (data_amount - train_size) // 2

indicies = torch.randperm(len(metadata_subsampled))
train_indices = indicies[:train_size]
val_indicies = indicies[train_size:train_size+val_size]
test_indicies = indicies[train_size+val_size:train_size+val_size+test_size]

# Checking there are not overlapping incdicies
#print(sum(np.isin(train_indices.numpy() , [val_indicies.numpy(), test_indicies.numpy()])))
#print(sum(np.isin(val_indicies.numpy() , [train_indices.numpy(), test_indicies.numpy()])))
#print(sum(np.isin(test_indicies.numpy() , [train_indices.numpy(), val_indicies.numpy()])))

training_set = torch.utils.data.Subset(train_set, train_indices.tolist())
val_set = torch.utils.data.Subset(train_set, val_indicies.tolist())
testing_set = torch.utils.data.Subset(train_set, test_indicies.tolist())

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

print(len(training_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=128, out_features=2*latent_features), # <- note the 2*latent_features
        )
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2*self.observation_features)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        #px_output = self.decoder(z)
        #px_output = px_output.view(-1, *self.input_shape) # reshape the output
        
        #return Bernoulli(logits=px_output, validate_args=False)
        
        px_output = self.decoder(z)
        mu, log_sigma = px_output.chunk(2, dim=-1)
        print(f"mu: {mu.shape}")
        print(f"log_sigma: {log_sigma.shape}")
        
        #mu = mu.view(-1, *self.input_shape)
        #log_sigma = log_sigma.view(-1, *self.input_shape)
        return ReparameterizedDiagonalGaussianWithSigmoid(mu, log_sigma) # Sample from the Normal distribution
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample() # qz.mu for test
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


latent_features = 10
vae = VariationalAutoencoder(images[0].shape, latent_features)
loss_fn = nn.MSELoss(reduction='none')
print(vae)

n_classes = 13
batches, height, width = x.shape


class Model(nn.Module):

    def __init__(self, n_classes):
        super(Model, self).__init__()
        activation_fn = nn.ReLU

        
        
        self.net = nn.Sequential(
          
            nn.Linear(256, 512),
            activation_fn(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1024),
            activation_fn(),
            nn.Linear(1024, 300),
            activation_fn(),
            nn.Dropout(p=0.1),
            nn.Linear(300, 100),
            activation_fn(),
            nn.Linear(100, n_classes)
            #nn.Flatten()
        )
        
        
    def forward(self, x):
        # Your code here!
        return self.net(x)
         
        
model = Model(n_classes)
print(model)

loss_fn = nn.CrossEntropyLoss()  # Your code here!
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Your code here!

num_epochs = 3
validation_every_steps = 50
model_latent = torch.load('new_vae.model')

step = 0
model.train()

train_accuracies = []
valid_accuracies = []
        
for epoch in range(num_epochs):
    
    train_accuracies_batches = []
    
    for inputs, targets in training_loader:
        
        
        # Forward pass, compute gradients, perform one training step.
        # Your code here!
        inputs = model_latent(inputs)['z']

        
        # Forward pass.
        output = model(inputs.detach())
        
        # Compute loss.
        loss = loss_fn(output.reshape(-1,13), targets)
        
        # Clean up gradients from the model.
        optimizer.zero_grad()
        
        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()
        
        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()
        
        # Increment step counter
        step += 1
        
        # Compute accuracy.
        predictions = output.reshape(-1,13).max(1)[1]
        train_accuracies_batches.append(accuracy(targets, predictions))
        
        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            
            train_accuracies_batches = []
        
            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    loss = loss_fn(output.reshape(-1,13), targets)

                    predictions = output.reshape(-1,13).max(1)[1]

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))

                model.train()
                
            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(dataset_test))
     
            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            print(f"             test accuracy: {valid_accuracies[-1]}")

print("Finished training.")

plt.figure()
plt.plot(range(50,step, 50), train_accuracies, 'r', range(50,step, 50), valid_accuracies, 'b')
plt.legend(['Train Accucary','Test Accuracy'])
plt.xlabel('steps'), plt.ylabel('Acc')
plt.savefig('classification_accuracy.png')
np.save('train_accuracies.npy',train_accuracies)
np.save('valid_accuracies.npy',valid_accuracies)
torch.save(model, result_dir+'classification.model')

