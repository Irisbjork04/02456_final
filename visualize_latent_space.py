import os
import random
import re
import time
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor, sigmoid
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision.utils import make_grid
from sklearn import metrics
import math 
from typing import *
from collections import defaultdict
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, Bernoulli
import torchvision.utils as vutils
from torch.nn.functional import softplus
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from umap import UMAP
from matplotlib import pyplot as plt
import seaborn as sns
#import plotly.express as px
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from model_VAE_plus import SingleCellDataset, PrintSize, Flatten, UnFlatten
from model_VAE_plus import ReparameterizedDiagonalGaussian
from model_VAE_plus import ReparameterizedDiagonalGaussianWithSigmoid
from model_VAE_plus import VariationalAutoencoder, VariationalInference

print(torch.__version__)

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

result_dir = 'result_visualizations/'
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


# Path to the model to be used
vae_model = '/work3/s193518/deep-learning-project-03/code/results_plus_VAE_1/vae_plus_final.model'
#vae_model = '/work3/s193518/deep-learning-project-03/code/results_vanilla_vae_2/vae.model'


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
num_epochs = 50

# Max patience for early stopping
max_patience = 50

# Learning rate for optimizers
lr = 1e-4

# Beta hyperparam for VAE loss
beta = 1.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# The value the DMSO category is downsampled to
downsample_value = 16000

# Amount of data used for training, validation and testing
data_prct = 1
train_prct = 0.95

# Number of classes
n_classes = 13

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

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, 
                                             shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

# Load a batch of images into memory
images, labels = next(iter(training_loader))
vae = VariationalAutoencoder(latent_features)
loss_fn = nn.MSELoss(reduction='none')
print(vae)

# Test with random input
vi_test = VariationalInference(beta=1)
print(images.shape)
loss, xhat, diagnostics, outputs = vi_test(vae, images)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")

# Evaluator: Variational Inference
vi = VariationalInference(beta=beta)

# Load pre-train VAE model
modelVAE = torch.load(vae_model)
modelVAE = modelVAE.to(device)

def one_hot(x, max_x):
    return torch.eye(max_x + 1)[x]

def get_latent_data(net, count=1000):

    latent_vectors = []
    latent_labels = []
    img_inputs = []
    rounds = count/100
    i = 0

    with torch.set_grad_enabled(False):
        dataset_loader = DataLoader(train_set, batch_size=100, shuffle=True)
        for inputs, labels in dataset_loader:
            
            inputs = inputs.to(device)
            #labels_one_hot = one_hot(labels, n_classes).to(device)

            loss, xhat, diagnostics, vae_outputs = vi(net, inputs)
            output = vae_outputs['z']

            if i == 0:
              latent_vectors = output.cpu()
              latent_labels = labels
              img_inputs = inputs
            else:
              latent_vectors = torch.cat((latent_vectors, output.cpu()), 0)
              latent_labels = torch.cat((latent_labels,labels), 0)
              img_inputs = torch.cat((img_inputs, inputs), 0)
            if i > rounds:
              break
            i+=1

    return img_inputs, latent_vectors, latent_labels

def plot_latent_2d(net, mode, count, method="tsne"):
    
    img_inputs, latent_vectors, latent_labels = get_latent_data(net=net, count=count)
    #latent_vectors = latent_vectors[latent_labels != 1]
    #latent_labels = latent_labels[latent_labels != 1]
    print(latent_vectors.shape)
    print(latent_labels.shape)
    
    fig, ax = plt.subplots(figsize=(10, 7))

    if method == "tsne":
        ax.set_title('t-SNE')
        coords = TSNE(n_components=2, random_state=42, perplexity=50.0, n_iter=5000,
                      learning_rate='auto', init="pca").fit_transform(latent_vectors)
    elif method == "pca":
        ax.set_title('pca')
        coords = PCA().fit_transform(latent_vectors)
    
    if mode == 'imgs':
        for image, (x, y) in zip(img_inputs.cpu(), coords):
            im = OffsetImage(image.reshape(68, 68), zoom=1, cmap='gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(coords)
        ax.autoscale()
    
    elif mode == 'dots':
        classes = latent_labels
        #plt.scatter(coords[:, 0], coords[:, 1], c=classes)
        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1],
            hue=classes,
            palette=sns.color_palette("hls", n_classes-1),
            legend="full",
            alpha=0.3
            )
        #plt.colorbar()
        for i in np.unique(latent_labels.numpy()):
            print(i)
            class_center = np.mean(coords[classes == i], axis=0)
            text = TextArea('{}'.format(classes_inv.get(i)))
            ab = AnnotationBbox(text, class_center, xycoords='data', frameon=True)
            ax.add_artist(ab)
    plt.savefig(result_dir + 'plus_VAE_TSNE_latent_space_perplexity_50_minus_class_1.png')
    plt.close()

""" def plot_latent_3d(net, mode, count, method="tsne"):
    
    img_inputs, latent_vectors, latent_labels = get_latent_data(net=net, count=count)
    
    fig, ax = plt.subplots(figsize=(10, 7))

    if method == "tsne":
        ax.set_title('t-SNE')
        coords = TSNE(n_components=3, random_state=42, 
                      learning_rate='auto', init="pca").fit_transform(latent_vectors)
    elif method == "pca":
        ax.set_title('pca')
        coords = PCA(n_components=3).fit_transform(latent_vectors)

    fig = px.scatter_3d(
    coords, x=0, y=1, z=2,
    color=latent_labels, labels=latent_labels
    )
    fig.update_traces(marker_size=8)

    plt.savefig(result_dir + 'plus_VAE_TSNE_latent_space_3d.png')
    plt.close() """

plot_latent_2d(net=modelVAE, mode='dots', count=10000, method="tsne")
#plot_latent_3d(net=modelVAE, mode='dots', count=1000, method="tsne")

