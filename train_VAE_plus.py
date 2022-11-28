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
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import pandas as pd
import math 
import torch
import torchvision.utils as vutils
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Initialize parameters
# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 68

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector
latent_features = 100

# Size of feature maps in VAE encoder and decoder
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Max patience for early stopping
max_patience = 40

# Learning rate for optimizers
lr_VAE = 0.01
lr_D = 1e-5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Beta2 hyperparam for VAE loss
beta2 = 2.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# The value the DMSO category is downsampled to
downsample_value = 16000

# Amount of data used for training, validation and testing
data_prct = 1
train_prct = 0.95

# Slope for delayed, linear, saturated schedulling of representation loss
slope  = 2500.0


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

    plt.savefig(name + '_real_images.png', bbox_inches='tight')
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

    plt.savefig(name + '_generated_images' + extra_name + '.png', bbox_inches='tight')
    plt.close()
    

def plot_curve(name, nll_val, x_label="epochs", y_label="nll"):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.close()


### Dataset

# Load metadata table
start_time = time.time()
metadata = pd.read_csv('/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv')
#metadata = pd.read_csv('/Users/mikkelrasmussen/mnt/deep_learning_project/data/metadata.csv', engine="pyarrow")
print("pd.read_csv wiht pyarrow took %s seconds" % (time.time() - start_time))

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


# The loaders perform the actual work
#images_folder = "/Users/mikkelrasmussen/mnt/deep_learning_project/data/singh_cp_pipeline_singlecell_images"
images_folder = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/'
train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x/x.max()),
     #transforms.Lambda(lambda x: x/torch.amax(x, dim=(0, 1))),
     #transforms.Lambda(lambda x: torch.flatten(x))
    ]
)

train_set = SingleCellDataset(images_folder=images_folder, 
                              annotation_file=metadata_subsampled, 
                              transform=train_transforms,
                              class_map=classes)

# Define the size of the train, validation and test datasets
data_amount = int(len(metadata_subsampled) * data_prct)
train_size = int(train_prct * data_amount)
val_size = (data_amount - train_size) // 2
test_size = (data_amount - train_size) // 2

indicies = torch.randperm(len(metadata_subsampled))
train_indices = indicies[:train_size]
val_indicies = indicies[train_size:train_size+val_size]
test_indicies = indicies[train_size+val_size:train_size+val_size+test_size]

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
    
    def __init__(self, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_features = latent_features
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.encoder = nn.Sequential(
            # W2=(W1−K+2P)/S]+1=[(68 - 6 + 2*0)/2]+1 = 32
            nn.Conv2d(in_channels=nc, 
                      out_channels=ngf, 
                      kernel_size=6, 
                      stride=2, 
                      padding=0, 
                      bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            PrintSize(),
            # W2=(W1−K+2P)/S]+1=[(32 - 4 + 2*1)/2]+1 = 16
            nn.Conv2d(in_channels=ngf, 
                      out_channels=ngf * 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            PrintSize(),
            # W2=(W1−K+2P)/S]+1=[(16 - 4 + 2*1)/2]+1 = 8
            nn.Conv2d(in_channels=ngf * 2, 
                      out_channels=ngf * 4, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            PrintSize(),
            # W2=(W1−K+2P)/S]+1=[(8 - 4 + 2*1)/2]+1 = 4
            nn.Conv2d(in_channels=ngf * 4, 
                      out_channels=ngf * 8, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            PrintSize(),
            # W2=(W1−K+2P)/S]+1=[(4 - 4 + 2*0)/2]+1 = 1
            nn.Conv2d(in_channels=ngf * 8, 
                      out_channels=latent_features*2,
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            PrintSize(),
            Flatten(),
            nn.Linear(in_features=latent_features*2, 
                      out_features=2*latent_features),
            PrintSize()
                    
        )
        
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            
            UnFlatten(),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=latent_features, 
                               out_channels=ngf * 8, 
                               kernel_size=4, 
                               stride=1, 
                               padding=0, 
                               bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            PrintSize(),
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (1-1)*1-2*0+1*(4-1)+0+1 = 4
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, 
                               out_channels=ngf * 4, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            PrintSize(),
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (4-1)*2-2*1+1*(4-1)+0+1 = 8
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 4, 
                               out_channels=ngf * 2, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            PrintSize(),
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (8-1)*2-2*1+1*(4-1)+0+1 = 16
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf * 2, 
                               out_channels=ngf, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            PrintSize(),
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (16-1)*2-2*1+1*(4-1)+0+1 = 32
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf, 
                               out_channels=nc * 2, 
                               kernel_size=8, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (32-1)*2-2*1+1*(8-1)+0+1 = 68
            # state size. (nc) x 68 x 68
            PrintSize()
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


vae = VariationalAutoencoder(latent_features)
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
    def __init__(self):
        super(Discriminator, self).__init__()
    
        # Defining activation function
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Define batchnorms
        self.batchnorm1 = nn.BatchNorm2d(ndf * 2)
        self.batchnorm2 = nn.BatchNorm2d(ndf * 4)
        self.batchnorm3 = nn.BatchNorm2d(ndf * 8)
        
        # CNN Layers
        self.conv_1 = nn.Conv2d(in_channels=nc,
                                out_channels=ndf,
                                kernel_size=6,
                                stride=2,
                                padding=0,
                                bias=False)
        
        self.conv_2 = nn.Conv2d(in_channels=ndf,
                                out_channels=ndf * 2,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False)

        self.conv_3 = nn.Conv2d(in_channels=ndf * 2,
                                out_channels=ndf * 4,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False)
        
        self.conv_4 = nn.Conv2d(in_channels=ndf * 4,
                                out_channels=ndf * 8,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False)
        
        self.conv_out = nn.Conv2d(in_channels=ndf * 8,
                                  out_channels=1,
                                  kernel_size=4,
                                  stride=1,
                                  padding=0,
                                  bias=False)
        
        # Max-pooling layer  # W2=(W1−F)/S+1]
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2))
        

    def forward(self, x_img):
        
        ## Convolutional layers ##        
        # Layer 1 [hw_in=68, hw_out=32]: INPUT LAYER
        x_img = self.conv_1(x_img) # W2=(W1−K+2P)/S]+1=[(68 - 6 + 2*0)/2]+1 = 32
        x_img = self.activation(x_img)
        x_img_1 = x_img
        
        # Layer 2 [hw_in=32, hw_out=16]
        x_img = self.conv_2(x_img) # W2=(W1−K+2P)/S]+1=((32 - 4 + 2*1)/2)+1 = 16
        x_img = self.batchnorm1(x_img)
        x_img = self.activation(x_img)
        x_img_2 = x_img

        # Layer 3 [hw_in=16, hw_out=8]
        x_img = self.conv_3(x_img) # W2=(W1−K+2P)/S]+1=((16 - 4 + 2*1)/2)+1 = 8
        x_img = self.batchnorm2(x_img)
        x_img = self.activation(x_img)
        x_img_3 = x_img

        # Layer 4 [hw_in=8, hw_out=4]: 
        x_img = self.conv_4(x_img) # W2=(W1−K+2P)/S]+1=((8 - 4 + 2*1)/2)+1 = 4
        x_img = self.batchnorm3(x_img)
        x_img = self.activation(x_img)
        x_img_4 = x_img
        
        # Layer 5 [hw_in=4, hw_out=1]:
        x_img = self.conv_out(x_img) # W2=(W1−K+2P)/S]+1=((4-4+ 2*0)/1)+1 = 1

        x_img = self.sigmoid(x_img)

        intermediate_rep = [x_img_1, x_img_2, x_img_3, x_img_4]
        output = x_img
        
        return output, intermediate_rep

discrim_test = Discriminator()
print(discrim_test)

output, intermediate_rep = discrim_test(images)
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
vae = VariationalAutoencoder(latent_features)

# Evaluator: Variational Inference
vi = VariationalInference(beta=beta2)

# Discriminator
discrim = Discriminator()
#discrim.apply(initialize_weights)

# The Adam optimizer works really well with VAEs.
#vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
#discriminator_optim = torch.optim.SGD(discrim.parameters(), lr=1e-5)

# Setup Adam optimizers for both G and D
#discriminator_optim = torch.optim.Adam(discrim.parameters(), lr=lr_D, betas=(beta1, 0.999))
discriminator_optim = torch.optim.SGD(discrim.parameters(), lr=lr_D)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=lr_VAE, betas=(beta1, 0.999))

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

def schedule_weight(delay):
    """ Defines a delayed, linear, saturated schedulling function.
    """
    step_norm = max(0.0, step - delay)
    w = step_norm / slope
    w = max(0.0, min(1.0, w)) #-- Bounded weight
    return w

epoch = 0
best_nll = 1000000.
patience = 0
nll_val = []
loss_repr_func = nn.MSELoss()
bce_loss = nn.BCELoss()

# Lists to keep track of progress
img_list = []
D_losses = []
VAE_losses = []
representation_loss = []
discriminator_real_loss = []
discriminator_fake_loss = []
discriminator_avg_loss = []
discriminator_sum_loss = []
total_loss_list = []
real_label = 1.0
fake_label = 0.0

name = 'vae_plus'
result_dir = 'results_plus_updated/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)

# move the model to the device
vae = vae.to(device)
discrim = discrim.to(device)

step = 0

# training..
while epoch < num_epochs:
    
    epoch += 1
    training_epoch_data = defaultdict(list)
    batch_discrim_avg_loss = []
    batch_discrim_sum_loss = []
    batch_discrim_real_loss = []
    batch_discrim_fake_loss = []
    batch_total_loss = []
    batch_repr_loss = []
    vae.train()
    discrim.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for i, (x, y) in enumerate(training_loader, 0):
            
            step += 1

            ############################
            # (Step 0) Prepare data:
            ###########################
            tmp_batch_size = x.size(0)
            b1, b2 = torch.split(x, split_size_or_sections=tmp_batch_size//2)
            batch_size_half = b1.size(0)
            
            b1 = b1.to(device)
            b2 = b2.to(device)
            
            # Reshape to in order to be used as input to conv layers
            b1_reshaped = b1.reshape(batch_size_half, 3, 68, 68)
            b2_reshaped = b2.reshape(batch_size_half, 3, 68, 68)

            
            ############################
            # (Step 1) Update Discriminator network:
            ###########################
            ## Train with all-real batch
            discriminator_optim.zero_grad()
            
            # Format labels
            label = torch.full((batch_size_half,), real_label,
                                dtype=torch.float, device=device)
            
            # Forward pass real batch through D
            b2_reshaped = b2.reshape(batch_size_half, 3, 68, 68)
            output_real_b2, inter_repr_real_b2 = discrim(b2_reshaped)
            output_real_b2 = output_real_b2.view(-1)
            
            # Calculate loss on all-real batch
            errD_real = bce_loss(output_real_b2, label)
            
            # Calculate gradients for D in backward pass
            errD_real.backward()

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

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            #print((errD_real/(errD_real + errD_fake)).item())
            #print(f"errD: {errD.item()}")

            # Update weights
            discriminator_optim.step()
            
            ############################
            # (Step 2) Update VAE network:
            ###########################

            vae_optimizer.zero_grad()

            # Pass real images from batch 1 (b1) through VAE:
            #loss_elbo, xhat, diagnostics, outputs = vi(vae, b1)
            #xhat_reshaped = xhat.reshape(batch_size_half, 3, 68, 68).to(device)
            
            # Pass x-hat (reconstructions) through Discriminator
            # without calculating gradients
            output_hat, inter_repr_fake = discrim(xhat_reshaped)
            
            # Pass real images from batch 1 (b1) through Discriminator
            # without calculating gradients
            output_real_b1, inter_repr_real_b1 = discrim(b1_reshaped)
            
            # Calculate the loss between the representations
            # of the real images and the reconstructions 
            loss_repr = 0
            loss_repr_list = []

            delays = [slope * (k+1) for k in range(len(inter_repr_fake))]

            for i, (repr_fake, repr_real) in enumerate(zip(inter_repr_fake, inter_repr_real_b1)):
                loss_batch = loss_repr_func(repr_fake, repr_real)
                loss_repr_list.append(loss_batch)

                loss_weight = schedule_weight(delays[i])
                loss_repr += loss_weight * loss_batch #-- Schedule-based weighted average
            
            loss_total = loss_elbo + loss_repr
            
            # Backpropagate the gradients for the VAE
            loss_total.backward()
            vae_optimizer.step()
                        
            # gather data for the current bach
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]
            
            batch_discrim_real_loss.append(errD_real.item())
            batch_discrim_fake_loss.append(errD_fake.item())
            batch_discrim_avg_loss.append((errD_real/(errD_real + errD_fake)).item())
            batch_discrim_sum_loss.append(errD.item())   
            batch_repr_loss.append(loss_repr.item())
            batch_total_loss.append(loss_total.item()) 

            # Output training stats
            if step % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_VAE: %.4f'
                  % (epoch, num_epochs, i, len(training_loader),
                     errD.item(), loss_total.item()))
                print(f"Training loss: {loss_total}")

            
            # Save Losses for plotting later
            D_losses.append(errD.item())
            VAE_losses.append(loss_total.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (step % 500 == 0) or ((epoch == num_epochs-1) and (i == len(training_loader)-1)):
                with torch.no_grad():
                    loss_elbo, xhat, diagnostics, outputs = vi(vae, b1)
                    xhat = xhat.detach().cpu()
                img_list.append(vutils.make_grid(xhat, padding=2, normalize=True))


    
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
    
    discriminator_real_loss.append(np.mean(batch_discrim_real_loss))
    discriminator_fake_loss.append(np.mean(batch_discrim_fake_loss))
    discriminator_avg_loss.append(np.mean(batch_discrim_avg_loss))
    discriminator_sum_loss.append(np.mean(batch_discrim_sum_loss))
    representation_loss.append(np.mean(batch_repr_loss))
    total_loss_list.append(np.mean(batch_total_loss))
        
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
    # make_vae_plots(vae, x, y, outputs, training_data, validation_data)

    if epoch == 1:
            print('saved!')
            torch.save(vae, result_dir + name + '.model')
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
               save_img= result_dir + "vae_out.png", save=True)

# Evaluation
test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(str(test_loss))
f.close()

samples_real(result_dir + name, test_loader)

plt.figure(figsize=(10,5))
plt.title("VAE and Discriminator Loss During Training")
plt.plot(VAE_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir + 'VAE_and_discriminator_training_loss.png', bbox_inches='tight')

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=60)
ani.save(result_dir + 'VAE_progression.mp4', writer=writervideo)
plt.close()

plot_curve(result_dir + name + "_nll_val_curve", nll_val)
plot_curve(result_dir + name + "_discriminator_real_loss", discriminator_real_loss, y_label="discriminator_real_loss")
plot_curve(result_dir + name + "_discriminator_fake_loss", discriminator_fake_loss, y_label="discriminator_fake_loss")
plot_curve(result_dir + name + "_discriminator_avg_loss", discriminator_avg_loss, y_label="discriminator_avg_loss")
plot_curve(result_dir + name + "_discriminator_sum_loss", discriminator_sum_loss, y_label="discriminator_sum_loss")
plot_curve(result_dir + name + "_representation_loss", representation_loss, y_label="representation_loss")
plot_curve(result_dir + name + "_total_loss", total_loss_list, y_label="total_loss")

