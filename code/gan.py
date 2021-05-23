import numpy as np
import torch
from torch import nn, optim
from facenet_pytorch import InceptionResnetV1


class Generator(nn.Module):
    def __init__(self, learning_rate=1e-3):
        # initialize
        super(Generator, self).__init__()
        self.input_dim = 100
        self.output_dim = (64, 64, 3)
        self.learning_rate = learning_rate

        # network layers
        # input: 100-dim latent vector
        # output: 64x64x3 RGB image
        self.h1 = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            )
        self.output = nn.Sequential(
            nn.Linear(16, np.prod(self.output_dim)),
            nn.Sigmoid(),
            )

        # optimization
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, latent_=None):
        latent = latent_ if latent_ is not None else torch.rand(1, 100)
        layer1 = self.h1(latent)
        output = self.output(layer1)
        return torch.reshape(output, self.output_dim)


class Discriminator(nn.Module):
    def __init__(self, learning_rate=1e-3):
        # initialize
        super(Discriminator, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, img):
        return self.model(img)


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        # initialize
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.unmasked_embeddings = None

    def compute_unmasked_embeddings(self, unmasked_faces):
        '''Compute and store database of embeddings for unmasked faces'''
        self.unmasked_embeddings = self.discriminator(unmasked_faces)

    def forward(self, masked_faces, masked_ids):
        '''One forward propagation of the GAN for a batch of faces'''
        # generate mask design from latent sample
        mask = self.generator()

        # project mask design onto images
        generator_faces = None

        # get embeddings from discriminator
        masked_embeddings = self.discriminator(masked_faces)
        generator_embeddings = self.discriminator(generator_faces)

        # determine distances from known, unmasked embeddings
        masked_distances = np.linalg.norm(self.unmasked_embeddings - masked_embeddings)  # NOTE: may need axis argument
        generator_distances = np.linalg.norm(self.unmasked_embeddings - generator_embeddings)  # NOTE: may need axis argument

        # determine identity of masked faces
        masked_identities = np.argmax(masked_distances)
        generator_identities = np.argmax(generator_distances)

        # return discriminator's classification accuracies
        # NOTE: need to be returning something the generator can train on, e.g. softmax probs?
        return np.mean(masked_identities == masked_ids), np.mean(generator_identities == masked_ids)






