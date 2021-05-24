import numpy as np
import torch
from torch import nn, optim
from facenet_pytorch import InceptionResnetV1
import pdb


class Generator(nn.Module):
    def __init__(self, learning_rate=1e-3):
        # initialize
        super(Generator, self).__init__()
        self.input_dim = 100
        self.output_dim = (128, 128, 3)
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

        # loss (cross entropy)
        self.criterion = nn.CrossEntropyLoss()

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

    def evaluate(self, query, target):
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
        accuracy = sum(classifications == torch.Tensor([ i for i in range(num_queries) ])) / num_queries
        return accuracy


class GAN():
    def __init__(self, generator, projector, discriminator, learning_rate=1e-3):
        '''
        Initialize GAN

        generator: generative network for producing mask designs (None -> 128x128 RGB facemask)
        projector: supervised network for projecting mask designs onto images of faces (128x128 RBG image -> 128x128 transparency mask)
        discriminator: facial recognition network for id-ing faces (128x128 RGB image -> 512-dimensional vector embedding)
        '''
        super(GAN, self).__init__()
        self.generator = generator
        self.projector = projector
        self.discriminator = discriminator
        self.unmasked_embeddings = None

        # train optimizer and loss on generator alone
        self.optimizer = self.generator.optimizer
        self.criterion = self.generator.criterion

    def compute_unmasked_embeddings(self, unmasked_faces):
        '''Compute and store database of embeddings for unmasked faces'''
        self.unmasked_embeddings = self.discriminator(unmasked_faces)

    def project_mask(self, facemask, masked_faces, process=False):
        transparency_masks = self.projector.predict(masked_faces, process=process)
        if facemask.shape[2] == 3:
            facemask = torch.transpose(facemask, 2, 1)
            facemask = torch.transpose(facemask, 1, 0)
        facemask = facemask.unsqueeze(0)
        generator_faces = (masked_faces + .5) * transparency_masks + facemask * (1 - transparency_masks)
        return generator_faces

    def one_hot(self, ids):
        classes = torch.zeros((self.unmasked_embeddings.shape[0], len(ids)))
        for i in range(classes.shape[0]):
            classes[i,ids[i]] = 1
        return classes

    def forward(self, masked_faces):
        '''One forward propagation of the GAN for a batch of faces'''
        # generate mask design from latent sample
        mask = self.generator()

        # project mask design onto images
        generator_faces = self.project_mask(mask, masked_faces) - .5

        # get embeddings from discriminator
        # masked_embeddings = self.discriminator(masked_faces)
        generator_embeddings = self.discriminator(generator_faces)

        # determine distances from known, unmasked embeddings
        # masked_distances = np.linalg.norm(batch_unmasked_embeddings - masked_embeddings)  # NOTE: may need axis argument
        # pdb.set_trace()
        difference = self.unmasked_embeddings.unsqueeze(0) - generator_embeddings.unsqueeze(1)
        generator_distances = torch.linalg.norm(difference, dim=2)

        # determine identity of masked faces
        # masked_identities = np.argmax(masked_distances)
        generator_identities = torch.argmax(generator_distances)

        # return softmax of distances
        # masked_softmax = nn.functional.softmax(masked_distances, dim=1)
        # generator_softmax = nn.functional.softmax(generator_distances, dim=1)
        return generator_distances

        # return discriminator's classification accuracies
        # NOTE: need to be returning something the generator can train on, e.g. softmax probs?
        # return np.mean(masked_identities == masked_ids), np.mean(generator_identities == masked_ids)

    def shuffle_data(self, inputs, outputs):
        n_examples = outputs.shape[0]
        shuffled_indices = torch.randperm(n_examples)
        inputs = inputs[shuffled_indices,:,:,:]
        outputs = outputs[shuffled_indices]
        return inputs, outputs

    def batch_data(self, inputs, outputs, batch_size=16):
        n_examples = outputs.shape[0]
        return [ (inputs[batch_size * i:batch_size * (i+1),:,:,:],
                  outputs[batch_size * i:batch_size * (i+1)])
                 for i in range(n_examples // batch_size) ]

    def train_batch(self, batch):
        inputs, correct_ids = batch
        generator_outputs = self.forward(inputs)
        self.optimizer.zero_grad()
        # pdb.set_trace()
        loss = self.criterion(generator_outputs, correct_ids)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return float(loss) / generator_outputs.shape[0]

    def fit(self, inputs, ids, num_epochs=10):
        # set model to train mode
        self.generator.train()

        # train over desired number of epochs
        epoch_loss = 0
        for epoch in range(1, num_epochs + 1):
            # sort data into minibatches
            inputs, ids = self.shuffle_data(inputs, ids)
            minibatches = self.batch_data(inputs, ids)

            # train on each minibatch
            epoch_loss = 0
            for batch in minibatches:
                epoch_loss += self.train_batch(batch)
            epoch_loss /= len(minibatches)

            # output loss
            if epoch % 10 == 0: print('Epoch {} loss: {}'.format(epoch, epoch_loss))

        # return training accuracy
        return epoch_loss

    def evaluate(self, inputs, correct_classes):
        # set model to eval mode
        self.generator.eval()

        # forward propagation
        batch_outputs = self.forward(inputs)

        # calculate classification accuracy
        output_classes = torch.argmin(batch_outputs, dim=1)
        accuracy = sum(output_classes == correct_classes) / len(correct_classes)
        return accuracy




