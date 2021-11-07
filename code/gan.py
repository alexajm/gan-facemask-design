import numpy as np
import torch
from torch import nn, optim
from facenet_pytorch import InceptionResnetV1
import pdb
import utils


class Generator(nn.Module):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-3, load_path=None):
        """Initialize generator for sampling facemask designs from latent space"""
        super(Generator, self).__init__()
        self.input_dim = 100
        self.output_dim = (3, 128, 128)
        self.learning_rate = learning_rate

        # network layers
        # input: 100-dim latent vector
        # output: 64x64x3 RGB image
        self.h1 = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
        )
        self.h2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(32, np.prod(self.output_dim)),
            nn.Sigmoid(),
        )

        # optimization
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # loss (cross entropy)
        self.criterion = nn.CrossEntropyLoss()

        # load prior weights
        if load_path:
            self.load_state_dict(torch.load(load_path))

    def forward(self, latent_=None):
        """Generate a facemask design from the sampled latent space"""
        latent = latent_ if latent_ is not None else torch.rand(1, 100)
        y = self.h1(latent)
        y = self.h2(y)
        output = self.output(y)
        output = torch.reshape(output, self.output_dim)


class Projector(nn.Module):
    def __init__(self, learning_rate=1e-3, load_path=None):
        """Initialize projector for projecting facemask designs onto images of individuals"""
        super(Projector, self).__init__()

        # hyperparameters
        channels = [3, 6, 12, 18, 24, 1]

        # layers
        # input: 128x128 RGB image
        # output: 128x128 transparency mask
        self.h1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.h2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.h3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.h4 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[3],
                out_channels=channels[4],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[4],
                out_channels=channels[5],
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        # optimizer (Adam)
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

        # loss (MSE)
        self.criterion = nn.MSELoss()

        # load prior weights
        if load_path:
            self.load_state_dict(torch.load(load_path))

    def forward(self, x):
        """One forward propagation of the projector for a batch of images"""
        y = self.h1(x)
        y = self.h2(y)
        y = self.h3(y)
        y = self.h4(y)
        y = self.output(y)
        return y

    def fit(self, inputs, outputs, num_epochs=10):
        """Train projector given masked faces as input and transparency masks as outputs"""
        self.train()
        return utils.fit(self, inputs, outputs, num_epochs=num_epochs)

    def predict(self, x, process=False):
        """Predict transparency mask for faces not trained on"""
        if len(x.shape) < 4:  # pre-process un-batched inputs
            x = torch.unsqueeze(x, 0)
        y = self.forward(x)
        if process:  # post-process outputs to enhance mask quality
            avg = torch.mean(y)
            y = (y > avg).type(torch.IntTensor)
        return y

    def evaluate(self, inputs, correct_outputs):
        """Evaluate loss of projector for given inputs"""
        # set model to eval mode
        self.eval()

        # forward propagation
        batch_outputs = self.forward(inputs)

        # return test loss
        loss = self.criterion(batch_outputs.float(), correct_outputs.float())
        n_examples = correct_outputs.shape[0]
        return float(loss) / n_examples

    def project_mask(self, facemask, masked_faces, process=False):
        """Project a facemask design onto images of masked individuals"""
        transparency_masks = self.predict(masked_faces, process=process)
        if masked_faces.device.type == "cuda":
            transparency_masks = transparency_masks.to(
                torch.device(
                    "{}:{}".format(masked_faces.device.type, masked_faces.device.index)
                )
            )
        if facemask.shape[2] == 3:
            facemask = torch.transpose(facemask, 2, 1)
            facemask = torch.transpose(facemask, 1, 0)
        facemask = facemask.unsqueeze(0)
        generator_faces = (masked_faces + 0.5) * transparency_masks + facemask * (
            1 - transparency_masks
        )
        return generator_faces


class Discriminator(nn.Module):
    def __init__(self, learning_rate=1e-3):
        """Initialize discriminator as pre-trained facial recognition model"""
        super(Discriminator, self).__init__()
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    def forward(self, img):
        """One forward propagation of the discriminator for a batch of images"""
        return self.model(img)

    def evaluate(self, query, target):
        """Evaluate classification accuracy of discriminator for given inputs"""
        # get embeddings
        target_embeddings = self.forward(target)
        query_embeddings = self.forward(query)

        # find distances between embeddings
        target_embeddings = torch.unsqueeze(target_embeddings, 1)
        query_embeddings = torch.unsqueeze(query_embeddings, 0)
        distances = torch.norm(target_embeddings - query_embeddings, dim=2)

        # classify queried images and return accuracy
        num_queries = query.shape[0]
        classifications = torch.argmin(distances, dim=0)
        accuracy = (
            sum(classifications == torch.Tensor([i for i in range(num_queries)]))
            / num_queries
        )
        return accuracy


class GAN:
    def __init__(self, generator, projector, discriminator, device=None):
        """
        Initialize GAN

        generator: generative network for producing mask designs (None -> 128x128 RGB facemask)
        projector: supervised network for projecting mask designs onto images of faces (128x128 RBG image -> 128x128 transparency mask)
        discriminator: facial recognition network for id-ing faces (128x128 RGB image -> 512-dimensional vector embedding)
        """
        super(GAN, self).__init__()
        self.generator = generator
        self.projector = projector
        self.discriminator = discriminator
        self.device = device
        self.unmasked_embeddings = None

        # send models to device if provided
        if device:
            self.generator.to(device)
            self.projector.to(device)
            self.discriminator.to(device)

        # train optimizer and loss on generator alone
        self.optimizer = self.generator.optimizer
        self.criterion = self.generator.criterion

    def compute_unmasked_embeddings(self, unmasked_faces):
        """Compute and store database of embeddings for unmasked faces"""
        if self.device:
            unmasked_faces = unmasked_faces.to(self.device)
        self.unmasked_embeddings = self.discriminator(unmasked_faces)
        if self.device:
            unmasked_faces = unmasked_faces.to(torch.device("cpu"))

    def generate_mask(self):
        """Generate a facemask design"""
        return self.generator()

    def project_mask(self, facemask, masked_faces, process=False):
        """Project a facemask design onto images of masked individuals"""
        return self.projector.project_mask(facemask, masked_faces, process=process)

    def one_hot(self, ids):
        """Generate one-hot vectors for given ids based on pre-computed embeddings"""
        classes = torch.zeros((self.unmasked_embeddings.shape[0], len(ids)))
        for i in range(classes.shape[0]):
            classes[i, ids[i]] = 1
        return classes

    def forward(self, masked_faces):
        """One forward propagation of the GAN for a batch of faces"""
        # generate mask design from latent sample
        mask = (
            self.generator(torch.rand(1, 100).to(self.device))
            if self.device
            else self.generator()
        )

        # project mask design onto images
        generator_faces = self.project_mask(mask, masked_faces) - 0.5

        # get embeddings from discriminator
        generator_embeddings = self.discriminator(generator_faces)

        # determine distances from known, unmasked embeddings
        difference = self.unmasked_embeddings.unsqueeze(
            0
        ) - generator_embeddings.unsqueeze(1)
        generator_distances = torch.linalg.norm(difference, dim=2)
        return generator_distances

    def fit(self, inputs, ids, num_epochs=10):
        """Train GAN given masked faces as input and their ids as labels"""
        self.generator.train()
        return utils.fit(self, inputs, ids, num_epochs=num_epochs)

    def evaluate(self, inputs, correct_classes, batch_size=64):
        """Evaluate classification accuracy of GAN for given inputs"""
        # set model to eval mode
        self.generator.eval()

        if self.device:
            # compute in batches
            accuracies = []
            minibatches = utils.batch_data(
                inputs, correct_classes, batch_size=batch_size
            )
            for batch in minibatches:
                # collect batch and send to CUDA
                batch_inputs, batch_correct_classes = batch
                batch_inputs = batch_inputs.to(self.device)
                batch_correct_classes = batch_correct_classes.to(self.device)

                # calculate classification accuracy
                batch_outputs = self.forward(batch_inputs)
                batch_output_classes = torch.argmin(batch_outputs, dim=1)
                accuracies.append(
                    sum(batch_output_classes == batch_correct_classes)
                    / len(batch_correct_classes)
                )

                # return batch to CPU
                batch_inputs = batch_inputs.to(torch.device("cpu"))
                batch_correct_classes = batch_correct_classes.to(torch.device("cpu"))
            return sum(accuracies) / len(accuracies)
        else:
            # forward propagation
            batch_outputs = self.forward(inputs)

            # calculate classification accuracy
            output_classes = torch.argmin(batch_outputs, dim=1)
            accuracy = sum(output_classes == correct_classes) / len(correct_classes)
            return accuracy

    def discriminator_evaluate(self, query, target, batch_size=64):
        """Evaluate classification accuracy of discriminator for given inputs"""
        if self.device:
            # compute in batches
            accuracies = []
            minibatches = utils.batch_data(query, target, batch_size=batch_size)
            for idx, batch in enumerate(minibatches):
                # collect batch and send to CUDA
                batch_queries, batch_targets = batch
                batch_queries = batch_queries.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # get embeddings
                target_embeddings = self.forward(batch_targets)
                query_embeddings = self.forward(batch_queries)

                # find distances between embeddings
                target_embeddings = torch.unsqueeze(target_embeddings, 1)
                query_embeddings = torch.unsqueeze(query_embeddings, 0)
                distances = torch.norm(target_embeddings - query_embeddings, dim=2)

                # classify queried images
                classifications = torch.argmin(distances, dim=0) + (idx * batch_size)
                id_range = torch.Tensor(
                    [(idx * batch_size) + i for i in range(batch_size)]
                ).to(self.device)
                accuracies.append(sum(classifications == id_range) / batch_size)

                # return batch to CPU
                batch_queries = batch_queries.to(torch.device("cpu"))
                batch_targets = batch_targets.to(torch.device("cpu"))
            return sum(accuracies) / len(accuracies)
        else:
            # get embeddings
            target_embeddings = self.forward(target)
            query_embeddings = self.forward(query)

            # find distances between embeddings
            target_embeddings = torch.unsqueeze(target_embeddings, 1)
            query_embeddings = torch.unsqueeze(query_embeddings, 0)
            distances = torch.norm(target_embeddings - query_embeddings, dim=2)

            # classify queried images and return accuracy
            num_queries = query.shape[0]
            classifications = torch.argmin(distances, dim=0)
            accuracy = (
                sum(classifications == torch.Tensor([i for i in range(num_queries)]))
                / num_queries
            )
            return accuracy
