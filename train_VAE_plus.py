# Load necessary packages
import os
from plotting import make_vae_plots
import re
import random
import time
from collections import defaultdict
from typing import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math 
import torch
from torch import nn, Tensor, sigmoid
from torch.nn.functional import softplus
from torch.distributions import Distribution, Bernoulli
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
sns.set_style("whitegrid")

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

### Evalutation and plotting functions

def evaluation(test_loader, name=None, model_best=None, epoch=None,
               device='cpu'):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')
        model_best = model_best.to(device)

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, (test_batch, test_target) in enumerate(test_loader):
        test_batch = test_batch.to(device)
        
        #loss_t = model_best.forward(test_batch, reduction='sum')
        loss_t, xhat, diagnostics, outputs = vi(model_best, test_batch)
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss

def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader))[0].detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.transpose(x[i].reshape((3, 68, 68)), (1, 2, 0))
        ax.imshow(plottable_image)
        ax.axis('off')

    plt.savefig(name + '_real_images.pdf', bbox_inches='tight')
    plt.close()
    

def samples_generated(name, data_loader, extra_name=''):
    x = next(iter(data_loader))[0].detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.cpu()
    model_best.eval()

    num_x = 4
    num_y = 4
    
    px = model_best.sample_from_prior(batch_size=num_x * num_y)['px']
    x_samples = px.sample()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x_samples[i], (3, 68, 68)).permute(1, 2, 0)
        ax.imshow(plottable_image)
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()
    

def plot_curve(name, nll_val, x_label="epochs", y_label="nll"):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(name + '.pdf', bbox_inches='tight')
    plt.close()


### Dataset

# Load metadata table
start_time = time.time()
metadata = pd.read_csv('/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv')
#metadata = pd.read_csv('/Users/mikkelrasmussen/mnt/deep_learning_project/data/metadata.csv', engine="pyarrow")
print("pd.read_csv wiht pyarrow took %s seconds" % (time.time() - start_time))

downsample_value = 16000

# DMSO category
DMSO_indx = metadata.index[metadata['moa'] == 'DMSO']
DMSO_drop_indices = np.random.choice(DMSO_indx, size=len(DMSO_indx) - downsample_value, replace=False)

# Microtubule stabilizers
#micro_indx = metadata.index[metadata['moa'] == 'Microtubule stabilizers']
#micro_drop_indices = np.random.choice(micro_indx, size=len(micro_indx) - downsample_value, replace=False)

#print(len(np.intersect1d(DMSO_drop_indices, micro_drop_indices)))
#all_drop_indices = np.concatenate((DMSO_drop_indices, micro_drop_indices))
all_drop_indices = DMSO_drop_indices

metadata_subsampled = metadata.drop(all_drop_indices, axis=0).reset_index()

# Map from class name to class index
classes = {index: name for name, index in enumerate(metadata["moa"].unique())}
classes_inv = {v: k for k, v in classes.items()}


# Dataloader class. Using the metadata table, images are sampled and 
# passed to VAE duing training
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

# Class implementing the reparameterization trick
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
        return -((z-self.mu)**2) / (2*self.sigma**2) - self.sigma.log() - math.log(math.sqrt(2 * math.pi))
        #return torch.distributions.Normal(self.mu, self.sigma).log_prob(z) #  alternative way

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


batch_size = 102

# The loaders perform the actual work
#images_folder = "/Users/mikkelrasmussen/mnt/deep_learning_project/data/singh_cp_pipeline_singlecell_images"
images_folder = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/'
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
data_prct = 0.1
train_prct = 0.95

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

# Load a batch of images into memory
images, labels = next(iter(training_loader))

# Variational autoencoder class

nbUnits = 32
latent_features = 256

class PrintSize(nn.Module):
    """Utility module to print current shape of a Tensor in Sequential, only at the first pass."""
    
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=latent_features):
        return input.view(input.size(0), size, 1, 1)

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
        #h_x = h_x.squeeze()
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
        mu, log_sigma = px_output.chunk(2, dim=1)
        
        return ReparameterizedDiagonalGaussianWithSigmoid(mu, log_sigma) # Sample from the Normal distribution
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        #x = x.view(x.size(0), -1)

        
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


vae = VariationalAutoencoder(images[0].shape, latent_features)
mse_loss = nn.MSELoss(reduction='none')
print(vae)

# Variational inference class
def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        #log_px = reduce(px.log_prob(x))
        xhat = px.rsample()
        log_px = - reduce(mse_loss(xhat, x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl # <- your code here
        beta_elbo = log_px - self.beta * kl # <- your code here
        
        # loss
        loss = - beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, xhat, diagnostics, outputs

vi_test = VariationalInference(beta=1)
loss, xhat, diagnostics, outputs = vi_test(vae, images)
print(f"{'xhat':6} | shape: {list(xhat.shape)}")
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")


# Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_shape:torch.Size, conv_channels:int, 
                 kernel_size:int, stride:int):
        super(Discriminator, self).__init__()
        
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.observation_features = np.prod(input_shape)

        # Defining activation function
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # Define batchnorms
        self.batchnorm1_2d = nn.BatchNorm2d(self.conv_channels)
        self.batchnorm2_2d = nn.BatchNorm2d(64)
        self.batchnorm3_2d = nn.BatchNorm2d(1)

        # Linear layers
        self.linear_first = nn.Linear(in_features=self.observation_features, out_features=256)
        self.linear_middle = nn.Linear(in_features=256, out_features=128)
        self.linear_last = nn.Linear(in_features=1, out_features=1)
        
        # CNN Layers
        self.conv_first = nn.Conv2d(in_channels=self.input_shape[1],
                             out_channels=self.conv_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride)
        
        self.conv_mid = nn.Conv2d(in_channels=self.conv_channels,
                               out_channels=self.conv_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride)
        
        self.conv_last = nn.Conv2d(in_channels=conv_channels,
                               out_channels=64,
                               kernel_size=self.kernel_size,
                               stride=self.stride)
        
        self.conv_out = nn.Conv2d(in_channels=64,
                               out_channels=1,
                               kernel_size=1,
                               stride=self.stride)
        
        # Max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2))
        

    def forward(self, x_img):
        
        ## Convolutional layers ##
        
        # Layer 0 [hw_in=68, hw_out=64>32]: INPUT LAYER
        x_img = self.conv_first(x_img) # W2=(W1−K+2P)/S]+1=[(68 - 5 + 2*0)/1]+1 = 64
        x_img = self.max_pool(x_img) # W2=(W1−F)/S+1]=[(64 - 2)/2] + 1 = 32
        x_img = self.activation(x_img)
        x_img = self.batchnorm1_2d(x_img)
        x_img_1 = x_img
        
        # Layer 1 [hw_in=32, hw_out=28>14]
        x_img = self.conv_mid(x_img) # W2=(W1−K+2P)/S]+1=[(32 - 5 + 2*0)/1]+1 = 28
        x_img = self.max_pool(x_img) # W2=(W1−F)/S+1]=[(28 - 2)/2] + 1 = 14
        x_img = self.activation(x_img)
        x_img = self.batchnorm1_2d(x_img)
        x_img_2 = x_img

        # Layer 2 [hw_in=14, hw_out=10>5]
        x_img = self.conv_mid(x_img) # W2=(W1−K+2P)/S]+1=[(14 - 5 + 2*0)/1]+1 = 10
        x_img = self.max_pool(x_img) # W2=(W1−F)/S+1]=[(10 - 2)/2] + 1 = 5
        x_img = self.activation(x_img)
        x_img = self.batchnorm1_2d(x_img)
        x_img_3 = x_img

        # Layer 3 [hw_in=5, hw_out=1]: 
        x_img = self.conv_last(x_img) # W2=(W1−K+2P)/S]+1=[(5 - 5 + 2*0)/1]+1 = 1
        x_img = self.activation(x_img)
        x_img = self.batchnorm2_2d(x_img)
        x_img_4 = x_img
        
        # Layer 4 [hw_in=1, hw_out=1]:
        x_img = self.conv_out(x_img) # W2=(W1−K+2P)/S]+1=[(1 - 1 + 2*0)/1]+1 = 1
        x_img = self.batchnorm3_2d(x_img)
        x_img_5 = x_img

        x_img = self.sigmoid(x_img)

        intermediate_rep = [x_img_1, x_img_2, x_img_3, x_img_4, x_img_5]
        output = x_img
        
        return output, intermediate_rep

discrim_test = Discriminator(images.reshape((102, 3, 68, 68)).shape, 
                             conv_channels=32, kernel_size=5, stride=1)
print(discrim_test)

output, intermediate_rep = discrim_test(images.reshape((102, 3, 68, 68)))
print(f"{'output':6} | shape: {list(output.shape)}")
for i, tensor in enumerate(intermediate_rep):
    print(f"x_img_{i+1} | shape: {list(tensor.shape)}")

# Weigth initialization
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# Define the models, evaluator and optimizer

# VAE
latent_features = 256
vae = VariationalAutoencoder(images[0].shape, latent_features)

# Evaluator: Variational Inference
beta = 2.0
vi = VariationalInference(beta=beta)

# Discriminator
stride = 1
kernel_size = 5
conv_channels = 32
input_size = (batch_size//2, 3, 68, 68)
discrim = Discriminator(input_size, 
                        conv_channels=conv_channels, 
                        kernel_size=kernel_size, stride=stride)
#discrim.apply(initialize_weights)

# The Adam optimizer works really well with VAEs.
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
discriminator_optim = torch.optim.SGD(discrim.parameters(), 1e-2, momentum=0.9)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

epoch = 0
num_epochs = 5
nll_val = []
best_nll = 1000000.
patience = 0
max_patience = 20
slope  = 2500.0
discriminator_layers = 5
loss_repr_func = nn.MSELoss()
bce_loss = nn.BCELoss()
representation_loss = []
discriminator_loss = []
total_loss = []
real_label = 1.0
fake_label = 0.0

name = 'vae_plus'
result_dir = 'results_plus/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)
discrim = discrim.to(device)

step = 0

# training..
while epoch < num_epochs:
    
    epoch += 1
    training_epoch_data = defaultdict(list)
    batch_discrim_loss = []
    batch_total_loss = []
    batch_repr_loss = []
    vae.train()
    discrim.train()
    D_losses = []
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in training_loader:

        if epoch < 4: 

            ### Step 0. Train discriminator alone for a few epochs.
            step += 1
            tmp_batch_size = x.size(0)
            b1, b2 = torch.split(x, split_size_or_sections=tmp_batch_size//2)
            batch_size_half = b1.size(0)
            
            b1 = b1.to(device)
            b2 = b2.to(device)
            
            # Reshape to in order to be used as input to conv layers
            b1_reshaped = b1.reshape(batch_size_half, 3, 68, 68)
            b2_reshaped = b2.reshape(batch_size_half, 3, 68, 68)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator_optim.zero_grad()
            
            # Format batch
            b2 = b2.to(device)
            batch_size_half = b2.size(0)
            label = torch.full((batch_size_half,), real_label, dtype=torch.float, device=device)
            
            # Forward pass real batch through D
            b2_reshaped = b2.reshape(batch_size_half, 3, 68, 68)
            output_real_b2, inter_repr_real_b2 = discrim(b2_reshaped)
            output_real_b2 = output_real_b2.view(-1)
            
            # Calculate loss on all-real batch
            errD_real = bce_loss(output_real_b2, label)
            print(f"errD_real: {errD_real.item()}")
            
            # Calculate gradients for D in backward pass
            errD_real.backward()
            #D_x = output_real_b2.mean().item()

            ## Train with all-fake batch
            ### Pass real images from batch 1 (b1) through VAE
            loss_elbo, xhat, diagnostics, outputs = vi(vae, b1)
            xhat_reshaped = xhat.reshape(batch_size_half, 3, 68, 68).to(device)
            label.fill_(fake_label)

            # Get the output from the discriminator
            output_hat, inter_repr_fake = discrim(xhat_reshaped.detach())
            output_hat = output_hat.view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = bce_loss(output_hat, label)
            print(f"errD_fake: {errD_fake.item()}")

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            #D_G_z1 = output_hat.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            #print((errD_real/(errD_real + errD_fake)).item())
            print(f"errD: {errD.item()}")
            D_losses.append(errD.item())

            # Update weights
            #errD.backward()
            discriminator_optim.step()


        else:
            step += 1
            tmp_batch_size = x.size(0)
            b1, b2 = torch.split(x, split_size_or_sections=tmp_batch_size//2)
            batch_size_half = b1.size(0)
            
            b1 = b1.to(device)
            b2 = b2.to(device)
            
            # Reshape to in order to be used as input to conv layers
            b1_reshaped = b1.reshape(batch_size_half, 3, 68, 68)
            b2_reshaped = b2.reshape(batch_size_half, 3, 68, 68)
            
            # True data is given label 1, while fake data is given label 0
            true_label = torch.ones(batch_size_half, 1).to(device)
            fake_label = torch.zeros(batch_size_half, 1).to(device)
            
            ### Step 1. Pass real images from batch 1 (b1) through VAE
            loss_elbo, xhat, diagnostics, outputs = vi(vae, b1)
            xhat_reshaped = xhat.reshape(batch_size_half, 3, 68, 68).to(device)
            
            ### Step 2. Pass x-hat (reconstructions) through Discriminator
            # without calculating gradients
            output_hat, inter_repr_fake = discrim(xhat_reshaped.detach())
            
            ### Step 3. Pass real images from batch 1 (b1) through Discriminator
            # without calculating gradients
            output_real_b1, inter_repr_real_b1 = discrim(b1_reshaped.detach())
            
            ### Step 4. Calculate the loss between the representations
            # of the real images and the reconstructions 
            loss_repr = 0
            loss_repr_list = []

            def schedule_weight(delay):
                """ Defines a delayed, linear, saturated schedulling function.
                """
                step_norm = max(0.0, step - delay)
                w = step_norm / slope
                w = max(0.0, min(1.0, w)) #-- Bounded weight
                return w

            delays = [slope * (k+1) for k in range(discriminator_layers)]

            for i, (repr_fake, repr_real) in enumerate(zip(inter_repr_fake, inter_repr_real_b1)):
                loss_batch = loss_repr_func(repr_fake, repr_real)
                loss_repr_list.append(loss_batch)

                loss_weight = schedule_weight(delays[i])
                loss_repr += loss_weight * loss_batch #-- Schedule-based weighted average
            
            loss_total = loss_elbo + loss_repr
            
            ### Step 5. Backpropagate the gradients for the VAE
            vae_optimizer.zero_grad()
            loss_total.backward()
            vae_optimizer.step()
            
            ### Step 6. Pass reconstrcutions (fake) and b2 real images through the discriminator,
            # backpropagate the errors and update the weights of the discriminator
            
            discriminator_optim.zero_grad()

            # Get the output from the discriminator
            output_hat, inter_repr_fake = discrim(xhat_reshaped.detach())
            output_real_b2, inter_repr_real_b2 = discrim(b2_reshaped)

            # Calculate the error and backpropagate 
            # For the real images (b2)
            output_real_b2 = output_real_b2.view(batch_size_half, 1)
            #output_real_b2_sigmoid = torch.sigmoid(output_real_b2)
            error_true = bce_loss(output_real_b2, true_label)
            error_true.backward()

            # For the reconstructions:
            output_hat = output_hat.view(batch_size_half, 1)
            #output_hat_sigmoid = torch.sigmoid(output_hat)
            error_fake = bce_loss(output_hat, fake_label)
            error_fake.backward()

            # Update weights
            discriminator_optim.step()
            
            # gather data for the current bach
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]
            
            batch_discrim_loss.append((error_true/(error_true + error_fake)).item())
            batch_repr_loss.append(loss_repr.item())
            batch_total_loss.append(loss_total.item())    

            print(f"Discriminator loss: {(error_true/(error_true + error_fake)).item()}")
            print(f"loss_repr_list: {loss_repr_list}")
            print(f"loss_elbo: {loss_elbo}")
            print(f"loss_repr: {loss_repr}")
            print(f"loss_total: {loss_total}")

    if epoch >= 4:      
        # gather data for the full epoch
        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]
        
        discriminator_loss.append(np.mean(batch_discrim_loss))
        representation_loss.append(np.mean(batch_repr_loss))
        total_loss.append(np.mean(batch_total_loss))
            
        # Evaluate on a single batch, do not propagate gradients
        with torch.no_grad():
            vae.eval()
            
            # Just load a single batch from the validation loader
            x, y = next(iter(val_loader))
            x = x.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss_val, xhat, diagnostics, outputs = vi(vae, x)
            print(f"loss_val: {loss_val}")
            nll_val.append(loss_val.cpu())  # save for plotting
            
            # gather data for the validation step
            for k, v in diagnostics.items():
                validation_data[k] += [v.mean().item()]
        
        # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
        make_vae_plots(vae, x, y, outputs, training_data, validation_data)

        if epoch == 1:
                print('saved!')
                torch.save(vae, result_dir + name + '_new.model')
                best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(vae, result_dir + name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(result_dir + name, val_loader, extra_name="_epoch_" + str(epoch))
            else:
                patience = patience + 1
            
        if patience > max_patience:
            print("Max patience reached! Performing early stopping!")
            break

make_vae_plots(vae, x, y, outputs, training_data, validation_data,
               save_img="results_plus/vae_out.png", save=True)

# Evaluation
test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(str(test_loss))
f.close()

samples_real(result_dir + name, test_loader)

plot_curve(result_dir + name + "_nll_val_curve", nll_val)
plot_curve(result_dir + name + "_discriminator_loss", discriminator_loss, y_label="discriminator_loss")
plot_curve(result_dir + name + "_representation_loss", representation_loss, y_label="representation_loss")
plot_curve(result_dir + name + "_total_loss", total_loss, y_label="total_loss")
