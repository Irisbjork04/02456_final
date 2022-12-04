
# Load necessary packages
import os
from plotting import make_vae_plots
import re
import random
import time
from typing import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import pandas as pd
import math 
import torch
from torch import nn, Tensor, sigmoid
from torch.nn.functional import softplus
import torchvision.utils as vutils
from torch.distributions import Distribution, Bernoulli
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
sns.set_style("whitegrid")

name = 'vae'
result_dir = 'results_vanilla_vae_3/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)

# Set random seed for reproducibility
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
f = open(result_dir + 'random_seed.txt', "w")
f.write(str(manualSeed))
f.close()
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

print("PyTorch Version {}" .format(torch.__version__))
print("Cuda Version {}" .format(torch.version.cuda))
print("CUDNN Version {}" .format(torch.backends.cudnn.version()))

torch.backends.cudnn.enabled = True

# Initialize parameters
# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 64

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
num_epochs = 90

# Max patience for early stopping
max_patience = 100

# Learning rate for optimizers
lr_VAE = 3e-4

# Beta hyperparam for VAE loss
beta = 1.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# The value the DMSO category is downsampled to
downsample_value = 16000

# Amount of data used for training, validation and testing
data_prct = 1
train_prct = 0.95

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Load metadata table
start_time = time.time()
metadata = pd.read_csv('/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv')
print("pd.read_csv wiht pyarrow took %s seconds" % (time.time() - start_time))

# Subsample the DMSO class
DMSO_indx = metadata.index[metadata['moa'] == 'DMSO']
DMSO_drop_indices = np.random.choice(DMSO_indx, size=260360, replace=False)
metadata_subsampled = metadata.drop(DMSO_drop_indices).reset_index()

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
        #return -((z-self.mu)**2) / (2*self.sigma**2) - self.sigma.log() - math.log(math.sqrt(2 * math.pi))
        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)


# A child class of the ReparameterizedDiagonalGaussian, which utilize the sigmoid on
# output samples. This is important when modeling the pixels with a Gaussian distribution
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


### Generate train, validation and test datasets
# Select the batch size to be used

# The loaders perform the actual work
images_folder = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/'
train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x/x.max())]
)

train_set = SingleCellDataset(images_folder=images_folder, 
                              annotation_file=metadata_subsampled, 
                              transform=train_transforms,
                              class_map=classes)

data_amount = int(len(metadata_subsampled) * data_prct)
train_size = int(train_prct * data_amount)
val_size = (data_amount - train_size) // 2
test_size = (data_amount - train_size) // 2

indicies = torch.randperm(len(metadata_subsampled))
train_indices = indicies[:train_size]
val_indicies = indicies[train_size:train_size+val_size]
test_indicies = indicies[train_size+val_size:train_size+val_size+test_size]

# Checking there are not overlapping incdicies
print(sum(np.isin(train_indices.numpy() , [val_indicies.numpy(), test_indicies.numpy()])))
print(sum(np.isin(val_indicies.numpy() , [train_indices.numpy(), test_indicies.numpy()])))
print(sum(np.isin(test_indicies.numpy() , [train_indices.numpy(), val_indicies.numpy()])))

training_set = torch.utils.data.Subset(train_set, train_indices.tolist())
val_set = torch.utils.data.Subset(train_set, val_indicies.tolist())
testing_set = torch.utils.data.Subset(train_set, test_indicies.tolist())

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, 
                                              shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

print(len(training_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))

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
            # W2=(W1−K+2P)/S]+1=[(32 - 4 + 2*1)/2]+1 = 16
            nn.Conv2d(in_channels=ngf, 
                      out_channels=ngf * 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # W2=(W1−K+2P)/S]+1=[(16 - 4 + 2*1)/2]+1 = 8
            nn.Conv2d(in_channels=ngf * 2, 
                      out_channels=ngf * 4, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # W2=(W1−K+2P)/S]+1=[(8 - 4 + 2*1)/2]+1 = 4
            nn.Conv2d(in_channels=ngf * 4, 
                      out_channels=ngf * 8, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # W2=(W1−K+2P)/S]+1=[(4 - 4 + 2*0)/2]+1 = 1
            nn.Conv2d(in_channels=ngf * 8, 
                      out_channels=latent_features*2,
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            Flatten(),
            nn.Linear(in_features=latent_features*2, 
                      out_features=2*latent_features),
                    
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
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (16-1)*2-2*1+1*(4-1)+0+1 = 32
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf, 
                               out_channels=nc * 2, 
                               kernel_size=8, 
                               stride=2, 
                               padding=1, 
                               bias=False)
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (32-1)*2-2*1+1*(8-1)+0+1 = 68
            # state size. (nc) x 68 x 68
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


# Load a batch of images into memory
images, labels = next(iter(training_loader))
vae = VariationalAutoencoder(latent_features)
loss_fn = nn.MSELoss(reduction='none')
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
        xhat = px.rsample()
        log_px = - reduce(loss_fn(xhat, x))
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

# Test with random input
vi_test = VariationalInference(beta=1)
print(images.shape)
loss, xhat, diagnostics, outputs = vi_test(vae, images)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")


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
        loss = loss + loss_t.cpu().item()
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
    model_best.to(device)
    model_best.eval()

    num_x = 4
    num_y = 4
    
    px = model_best.sample_from_prior(batch_size=num_x * num_y)['px']
    x_samples = px.sample()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x_samples[i].cpu(), (3, 68, 68)).permute(1, 2, 0)
        ax.imshow(plottable_image)
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.png', bbox_inches='tight')
    plt.close()
    

def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.png', bbox_inches='tight')
    plt.close()

name = 'vae'

from collections import defaultdict
# define the models, evaluator and optimizer

# VAE
vae = VariationalAutoencoder(latent_features)

# Evaluator: Variational Inference
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=lr_VAE)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

epoch = 0
nll_val = []
elbo_loss = []
kl_loss = []
log_px_loss = []
train_loss = []
best_nll = 1000000.
patience = 0
step = 0
train_img_list = []
val_img_list = []

train_fixed, train_label_fixed = next(iter(val_loader))
train_fixed_b1, train_fixed_b2 = torch.split(train_fixed, split_size_or_sections=batch_size//2)
train_fixed_b1 = train_fixed_b1.reshape(batch_size//2, 3, 68, 68)
train_fixed_b1 = train_fixed_b1.to(device)

val_fixed, val_label_fixed = next(iter(val_loader))
val_fixed_b1, val_fixed_b2 = torch.split(val_fixed, split_size_or_sections=batch_size//2)
val_fixed_b1 = val_fixed_b1.reshape(batch_size//2, 3, 68, 68)
val_fixed_b1 = val_fixed_b1.to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)

# training..
while epoch < num_epochs:
    epoch += 1
    print(f"{epoch} / {num_epochs}")
    training_epoch_data = defaultdict(list)
    vae.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for i, (x, y) in enumerate(training_loader):
        step += 1
        
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, xhat, diagnostics, outputs = vi(vae, x)

        elbo_loss.append(diagnostics['elbo'].mean().item())
        kl_loss.append(diagnostics['kl'].mean().item())
        log_px_loss.append(diagnostics['log_px'].mean().item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
        
        # Output training stats
        if step % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_VAE: %.4f'
                % (epoch, num_epochs, i, len(training_loader), loss.item()))

        # Check how the generator is doing by saving G's output on fixed_noise
        if (step % 100 == 0) or ((epoch == num_epochs-1) and (i == len(training_loader)-1)):
            with torch.no_grad():
                loss_elbo, xhat, diagnostics, outputs = vi(vae, train_fixed_b1)
                xhat = xhat.detach().cpu()
            train_img_list.append(vutils.make_grid(xhat, padding=2, normalize=True))

            with torch.no_grad():
                loss_elbo, xhat, diagnostics, outputs = vi(vae, val_fixed_b1)
                xhat = xhat.detach().cpu()
            val_img_list.append(vutils.make_grid(xhat, padding=2, normalize=True))
            
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
        
    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()
        
        # Just load a single batch from the validation loader
        x, y = next(iter(val_loader))
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss_val, xhat, diagnostics, outputs = vi(vae, x)
        nll_val.append(loss_val.cpu())  # save for plotting
        
        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
    
    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    if (step % 3000 == 0):
        filename = result_dir + f"vae_out_{epoch}.png"
        make_vae_plots(vae, x, y, outputs, training_data, validation_data, save_img = filename, save=True)

    if epoch == 1:
            print('saved!')
            torch.save(vae, result_dir + name + '.model')
            best_nll = loss_val
    else:
        samples_generated(result_dir + name, val_loader, extra_name="_epoch_" + str(epoch))
        if loss_val < best_nll:
            print('saved!')
            torch.save(vae, result_dir + name + '.model')
            best_nll = loss_val
            patience = 0
        else:
            patience = patience + 1
        
    if patience > max_patience:
        print("Max patience reached! Performing early stopping!")
        break

print('saved final model!')
torch.save(vae, result_dir + name + '_final.model')

make_vae_plots(vae, x, y, outputs, training_data, validation_data,
               save_img = result_dir + "vae_out_final.png", save=True)

# Evaluation
test_loss = evaluation(name=result_dir + name, test_loader=test_loader, device=device)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(str(test_loss))
f.close()

samples_real(result_dir + name, test_loader)

plot_curve(result_dir + name, nll_val)

plt.figure(figsize=(10,5))
plt.title("Elbo Loss During Training")
plt.plot(elbo_loss, label="Elbo")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir + 'elbo_training_loss.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
plt.title("KL Loss During Training")
plt.plot(elbo_loss, label="KL")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir + 'kl_training_loss.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
plt.title("Log(p(x|z)) Loss During Training")
plt.plot(elbo_loss, label="Log(p(x|z))")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir + 'log_px_training_loss.png', bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in train_img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=60)
ani.save(result_dir + 'train_VAE_progression.mp4', writer=writervideo)
plt.close()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in val_img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=60)
ani.save(result_dir + 'val_VAE_progression.mp4', writer=writervideo)
plt.close()