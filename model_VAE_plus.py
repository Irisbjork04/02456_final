import re
import os
import math
import numpy as np
from typing import *
import torch
from torch import nn, Tensor, sigmoid
from torch.distributions import Distribution
from torchvision.transforms import ToTensor


## Parameters
# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector
latent_features = 100

# Size of feature maps in VAE encoder and decoder
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Loss function
mse_loss = nn.MSELoss(reduction='none')

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

        """ self.encoder = nn.Sequential(
          # Layer 0 [hw_in=68, hw_out=64>32]: INPUT LAYER
            # W2=(W1−K+2P)/S]+1=[(68 - 5 + 2*0)/1]+1 = 64
            nn.Conv2d(in_channels=nc, 
                      out_channels=ngf, 
                      kernel_size=5, 
                      stride=1), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)), # W2=(W1−F)/S+1]=[(64 - 2)/2] + 1 = 32
            nn.LeakyReLU(),
            PrintSize(),

            # Layer 1 [hw_in=32, hw_out=28>14]
            # W2=(W1−K+2P)/S]+1=[(32 - 5 + 2*0)/1]+1 = 28
            nn.Conv2d(in_channels=ngf, 
                      out_channels=ngf, 
                      kernel_size=(5,5), 
                      stride=(1,1)), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)), # W2=(W1−F)/S+1]=[(28 - 2)/2] + 1 = 14
            nn.LeakyReLU(),
            PrintSize(),

            # Layer 2 [hw_in=14, hw_out=10>5]
            # W2=(W1−K+2P)/S]+1=[(14 - 5 + 2*0)/1]+1 = 10
            nn.Conv2d(in_channels=ngf, 
                      out_channels=ngf, 
                      kernel_size=(5,5), 
                      stride=(1,1)), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)), # W2=(W1−F)/S+1]=[(10 - 2)/2] + 1 = 5
            nn.LeakyReLU(),
            PrintSize(),

            # Layer 3 [hw_in=5, hw_out=1]: EMBEDDING
            # W2=(W1−K+2P)/S]+1=[(5 - 5 + 2*0)/1]+1 = 1
            nn.Conv2d(in_channels=ngf, 
                      out_channels=ngf, 
                      kernel_size=(5,5), 
                      stride=(1,1)), 
            PrintSize(),
            Flatten(),
            nn.Linear(in_features=ngf, out_features=2*latent_features),
            PrintSize(),
        )


        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(

            # Layer 3+1 [hw_in=1, hw_out=5<10]
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (1-1)*1-2*0+1*(4-1)+0+1 = 4
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=latent_features, 
                               out_channels=ngf, 
                               kernel_size=5, 
                               stride=1,
                               padding=0),
            nn.UpsamplingNearest2d(size=(10,10)),
            nn.LeakyReLU(),
            PrintSize(),

            # Layer 3+2 [hw_in=10<10+8, hw_out=14<28]
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (4-1)*2-2*1+1*(4-1)+0+1 = 8
            nn.ConvTranspose2d(in_channels=ngf, 
                               out_channels=ngf, 
                               kernel_size=5, 
                               stride=1,
                               padding=0),
            nn.UpsamplingNearest2d(size=(28,28)),
            nn.LeakyReLU(),
            PrintSize(),

            # Layer 3+3 [hw_in=28<28+8, hw_out=32<64]
            # (H−1)×S−2×P+D×(K−1)+OP+1 = (8-1)*2-2*1+1*(4-1)+0+1 = 16
            nn.ConvTranspose2d(in_channels=ngf, 
                               out_channels=ngf, 
                               kernel_size=5, 
                               stride=1,
                               padding=0),
            nn.UpsamplingNearest2d(size=(64,64)),
            nn.LeakyReLU(),
            PrintSize(),

             # (H−1)×S−2×P+D×(K−1)+OP+1 = (16-1)*2-2*1+1*(4-1)+0+1 = 32
            nn.ConvTranspose2d(in_channels=ngf, 
                               out_channels=ngf, 
                               kernel_size=5, 
                               stride=1,
                               padding=0),
            nn.UpsamplingNearest2d(size=(64,64)),
            nn.LeakyReLU(),
            PrintSize(),

            # Layer 3+4 [hw_in=64, hw_out=68]: FINAL ACTIVATION
            nn.ConvTranspose2d(in_channels=ngf, 
                      out_channels=2*nc, 
                      kernel_size=5, stride=1, padding=0)
        ) """
        
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
        

    def forward(self, x, validation=False) -> Dict[str, Any]:
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

""" class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        # Defining activation function
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Define batchnorms
        self.batchnorm = nn.BatchNorm2d(ndf)
        
        # CNN Layers
        self.conv_1 = nn.Conv2d(in_channels=nc,
                                out_channels=ndf,
                                kernel_size=5,
                                stride=1,
                                padding=0,
                                bias=False)
        
        self.conv_2 = nn.Conv2d(in_channels=ndf,
                                out_channels=ndf,
                                kernel_size=5,
                                stride=1,
                                padding=0,
                                bias=False)

        self.conv_3 = nn.Conv2d(in_channels=ndf,
                                out_channels=ndf,
                                kernel_size=5,
                                stride=1,
                                padding=0,
                                bias=False)
        
        self.conv_4 = nn.Conv2d(in_channels=ndf,
                                out_channels=ndf*2,
                                kernel_size=5,
                                stride=1,
                                padding=0,
                                bias=False)
        
        self.conv_out = nn.Conv2d(in_channels=ndf*2,
                                  out_channels=1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False)
        
        # Max-pooling layer  # W2=(W1−F)/S+1]
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2))
        

    def forward(self, x_img):
        
        ## Convolutional layers ##        
        # Layer 1 [hw_in=68, hw_out=32]: INPUT LAYER
        x_img = self.conv_1(x_img) # W2=(W1−K+2P)/S]+1=((68-5+2*0)/1)+1 = 64
        x_img = self.max_pool(x_img) # W2=[(W1−F)/S]+1 = ((64 - 2)/2) + 1 = 32
        x_img = self.batchnorm(x_img)
        x_img = self.activation(x_img)
        x_img_1 = x_img
        
        # Layer 2 [hw_in=32, hw_out=16]
        x_img = self.conv_2(x_img) # W2=(W1−K+2P)/S]+1=((32-5+2*0)/1)+1 = 28
        x_img = self.max_pool(x_img) # W2=[(W1−F)/S]+1 = ((28 - 2)/2) + 1 = 14
        x_img = self.batchnorm(x_img)
        x_img = self.activation(x_img)
        x_img_2 = x_img

        # Layer 3 [hw_in=16, hw_out=8]
        x_img = self.conv_3(x_img) # W2=(W1−K+2P)/S]+1=((14-5+2*0)/1)+1 = 10
        x_img = self.max_pool(x_img) # W2=[(W1−F)/S]+1 = ((10 - 2)/2) + 1 = 5
        x_img = self.batchnorm(x_img)
        x_img = self.activation(x_img)
        x_img_3 = x_img

        # Layer 4 [hw_in=8, hw_out=4]: 
        x_img = self.conv_4(x_img) # W2=(W1−K+2P)/S]+1=((5-5+2*0)/1)+1 = 1
        x_img = self.activation(x_img)
        #x_img_4 = x_img
        
        # Layer 5 [hw_in=4, hw_out=1]:
        x_img = self.conv_out(x_img) # W2=(W1−K+2P)/S]+1=((4-4+ 2*0)/1)+1 = 1

        x_img = self.sigmoid(x_img)

        intermediate_rep = [x_img_1, x_img_2, x_img_3]#, x_img_4]
        output = x_img
        
        return output, intermediate_rep """
