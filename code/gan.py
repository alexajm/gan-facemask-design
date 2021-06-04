import numpy as np
import torch
from torch import nn, optim
from facenet_pytorch import InceptionResnetV1
import pdb
import utils
import styles


class Generator(nn.Module):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-3, styles=[]):
        '''Initialize generator for sampling facemask designs from latent space'''
        super(Generator, self).__init__()
        self.input_dim = 100
        self.output_dim = (3, 128, 128)
        self.learning_rate = learning_rate
        self.styles = styles

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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # loss (cross entropy)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, latent_=None):
        '''Generate a facemask design from the sampled latent space'''
        latent = latent_ if latent_ is not None else torch.rand(1, 100)
        y = self.h1(latent)
        y = self.h2(y)
        output = self.output(y)
        output = torch.reshape(output, self.output_dim)

        # apply styles
        for style in self.styles:
            output = style(output)
        return output


class Projector(nn.Module):
    def __init__(self, learning_rate=1e-3, load_path=None):
        '''Initialize projector for projecting facemask designs onto images of individuals'''
        super(Projector, self).__init__()

        # hyperparameters
        channels = [3, 6, 12, 18, 24, 1]

        # layers
        # input: 128x128 RGB image
        # output: 128x128 transparency mask
        self.h1 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.h2 = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.h3 = nn.Sequential(
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.h4 = nn.Sequential(
            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=channels[4], out_channels=channels[5], kernel_size=3, padding=1),
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
        '''One forward propagation of the projector for a batch of images'''
        y = self.h1(x)
        y = self.h2(y)
        y = self.h3(y)
        y = self.h4(y)
        y = self.output(y)
        return y

    def fit(self, inputs, outputs, num_epochs=10):
        '''Train projector given masked faces as input and transparency masks as outputs'''
        self.train()
        return utils.fit(self, inputs, outputs, num_epochs=num_epochs)

    def predict(self, x, process=False):
        '''Predict transparency mask for faces not trained on'''
        if len(x.shape) < 4:  # pre-process un-batched inputs
            x = torch.unsqueeze(x, 0)
        y = self.forward(x)
        if process:  # post-process outputs to enhance mask quality
            avg = torch.mean(y)
            y = (y > avg).type(torch.IntTensor)
        return y

    def evaluate(self, inputs, correct_outputs):
        '''Evaluate loss of projector for given inputs'''
        # set model to eval mode
        self.eval()

        # forward propagation
        batch_outputs = self.forward(inputs)

        # return test loss
        loss = self.criterion(batch_outputs.float(), correct_outputs.float())
        n_examples = correct_outputs.shape[0]
        return float(loss) / n_examples

    def project_mask(self, facemask, masked_faces, process=False):
        '''Project a facemask design onto images of masked individuals'''
        transparency_masks = self.predict(masked_faces, process=process)
        if masked_faces.device.type == 'cuda': transparency_masks = transparency_masks.to(torch.device('{}:{}'.format(masked_faces.device.type, masked_faces.device.index)))
        if facemask.shape[2] == 3:
            facemask = torch.transpose(facemask, 2, 1)
            facemask = torch.transpose(facemask, 1, 0)
        facemask = facemask.unsqueeze(0)
        generator_faces = (masked_faces + .5) * transparency_masks + facemask * (1 - transparency_masks)
        return generator_faces


class Discriminator(nn.Module):
    def __init__(self, learning_rate=1e-3, regularization=1e-2):
        '''Initialize discriminator as pre-trained facial recognition model'''
        super(Discriminator, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

        # set trainable layer for competition against generator
        self.finetune = self.model.last_linear
        self.finetune.train()
        self.optimizer = optim.Adam(self.finetune.parameters(), learning_rate)
        # pdb.set_trace()
        self.criterion = lambda x, y : -1 * nn.CrossEntropyLoss()(x, y) + regularization * torch.sum(torch.abs(self.finetune.weight))

    def forward(self, img):
        '''One forward propagation of the discriminator for a batch of images'''
        return self.model(img)

    def evaluate(self, query, target):
        '''Evaluate classification accuracy of discriminator for given inputs'''
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
    def __init__(self, generator, projector, discriminator, device=None):
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
        '''Compute and store database of embeddings for unmasked faces'''
        if self.device: unmasked_faces = unmasked_faces.to(self.device)
        self.unmasked_embeddings = self.discriminator(unmasked_faces).detach()
        if self.device: unmasked_faces = unmasked_faces.to(torch.device('cpu'))

    def generate_mask(self):
        '''Generate a facemask design'''
        return self.generator()

    def project_mask(self, facemask, masked_faces, process=False):
        '''Project a facemask design onto images of masked individuals'''
        return self.projector.project_mask(facemask, masked_faces, process=process)

    def one_hot(self, ids):
        '''Generate one-hot vectors for given ids based on pre-computed embeddings'''
        classes = torch.zeros((self.unmasked_embeddings.shape[0], len(ids)))
        for i in range(classes.shape[0]):
            classes[i,ids[i]] = 1
        return classes

    def forward(self, masked_faces):
        '''One forward propagation of the GAN for a batch of faces'''
        # generate mask design from latent sample
        mask = self.generator(torch.rand(1, 100).to(self.device)) if self.device else self.generator()

        # project mask design onto images
        generator_faces = self.project_mask(mask, masked_faces) - .5

        # get embeddings from discriminator
        generator_embeddings = self.discriminator(generator_faces)

        # determine distances from known, unmasked embeddings
        difference = self.unmasked_embeddings.unsqueeze(0) - generator_embeddings.unsqueeze(1)
        generator_distances = torch.linalg.norm(difference, dim=2)
        return generator_distances

    def fit(self, masked_faces, unmasked_faces, correct_ids, num_epochs=10):
        # set models to train mode
        self.generator.train()
        self.discriminator.train()

        # fit on data
        gen_epoch_loss = 0
        dis_epoch_loss = 0
        for epoch in range(1, num_epochs + 1):
            # sort data into minibatches
            masked_faces, unmasked_faces, correct_ids = self.shuffle_data(masked_faces, unmasked_faces, correct_ids)
            minibatches = self.batch_data(masked_faces, unmasked_faces, correct_ids)

            # train on each minibatch
            gen_epoch_loss = 0
            dis_epoch_loss = 0
            for batch in minibatches:
                gen_loss, dis_loss = self.train_batch(batch)
                gen_epoch_loss += gen_loss
                dis_epoch_loss += dis_loss
            gen_epoch_loss /= len(minibatches)
            dis_epoch_loss /= len(minibatches)

            # output loss
            if epoch % 10 == 0: print('Epoch {} losses: \t\t{} \t\t{}'.format(epoch, gen_epoch_loss, dis_epoch_loss))

    def shuffle_data(self, masked, unmasked, outputs):
        '''Shuffle the first dimension of a set of input/output data'''
        n_examples = outputs.shape[0]
        shuffled_indices = torch.randperm(n_examples)
        masked = masked[shuffled_indices]
        unmasked = unmasked[shuffled_indices]
        outputs = outputs[shuffled_indices]
        return masked, unmasked, outputs


    def batch_data(self, masked, unmasked, outputs, batch_size=16):
        '''Convert full input/output pairs to a list of batched tuples'''
        n_examples = outputs.shape[0]
        return [ (masked[batch_size * i:batch_size * (i+1)],
                  unmasked[batch_size * i:batch_size * (i+1)],
                  outputs[batch_size * i:batch_size * (i+1)],)
                 for i in range(n_examples // batch_size) ]

    def train_batch(self, batch):
        '''Perform one iteration of model training given a single batch'''
        # send data to CUDA if necessary
        masked_faces, unmasked_faces, correct_ids = batch
        if self.device:
            masked_faces = masked_faces.to(self.device)
            unmasked_faces = unmasked_faces.to(self.device)
            correct_ids = correct_ids.to(self.device)

        # generate mask design from latent sample
        mask = self.generator(torch.rand(1, 100).to(self.device)) if self.device else self.generator()

        # project mask design onto images
        generator_faces = self.project_mask(mask, masked_faces) - .5

        # get embeddings from discriminator
        generator_embeddings = self.discriminator(generator_faces)

        # determine distances from known, unmasked embeddings
        difference = self.unmasked_embeddings.unsqueeze(0) - generator_embeddings.unsqueeze(1)
        generator_distances = torch.linalg.norm(difference, dim=2)

        # train generator
        self.generator.optimizer.zero_grad()
        generator_loss = self.generator.criterion(generator_distances, correct_ids)
        generator_loss.backward(retain_graph=True)
        self.generator.optimizer.step()

        # train discriminator (unmasked)
        new_unmasked_embeddings = self.discriminator(unmasked_faces)
        difference = self.unmasked_embeddings.unsqueeze(0) - new_unmasked_embeddings.unsqueeze(1)
        unmasked_distances = torch.linalg.norm(difference, dim=2)
        self.discriminator.optimizer.zero_grad()
        discriminator_loss = self.discriminator.criterion(unmasked_distances, correct_ids)
        discriminator_loss.backward(retain_graph=True)
        self.discriminator.optimizer.step()

        # train discriminator (masked)
        detached_faces = generator_faces.detach()
        detached_embeddings = self.discriminator(detached_faces)
        difference = self.unmasked_embeddings.unsqueeze(0) - detached_embeddings.unsqueeze(1)
        detached_distances = torch.linalg.norm(difference, dim=2)
        self.discriminator.optimizer.zero_grad()
        discriminator_loss = self.discriminator.criterion(detached_distances, correct_ids)
        discriminator_loss.backward(retain_graph=True)
        self.discriminator.optimizer.step()

        # return data to CPU if necessary
        if self.device:
            masked_faces = masked_faces.to(torch.device('cpu'))
            correct_ids = correct_ids.to(torch.device('cpu'))
        return float(generator_loss) / generator_distances.shape[0], float(discriminator_loss) / detached_distances.shape[0]

    def evaluate(self, inputs, correct_classes, batch_size=16):
        '''Evaluate classification accuracy of GAN for given inputs'''
        # set model to eval mode
        self.generator.eval()

        if self.device:
            # compute in batches
            accuracies = []
            minibatches = utils.batch_data(inputs, correct_classes, batch_size=batch_size)
            for batch in minibatches:
                # collect batch and send to CUDA
                batch_inputs, batch_correct_classes = batch
                batch_inputs = batch_inputs.to(self.device)
                batch_correct_classes = batch_correct_classes.to(self.device)

                # calculate classification accuracy
                batch_outputs = self.forward(batch_inputs)
                batch_output_classes = torch.argmin(batch_outputs, dim=1)
                accuracies.append(sum(batch_output_classes == batch_correct_classes) / len(batch_correct_classes))

                # return batch to CPU
                batch_inputs = batch_inputs.to(torch.device('cpu'))
                batch_correct_classes = batch_correct_classes.to(torch.device('cpu'))
            return sum(accuracies) / len(accuracies)
        else:
            # forward propagation
            batch_outputs = self.forward(inputs)

            # calculate classification accuracy
            output_classes = torch.argmin(batch_outputs, dim=1)
            accuracy = sum(output_classes == correct_classes) / len(correct_classes)
            return accuracy

    def discriminator_evaluate(self, query, target, batch_size=16):
        '''Evaluate classification accuracy of discriminator for given inputs'''
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
                id_range = torch.Tensor([ (idx * batch_size) + i for i in range(batch_size) ]).to(self.device)
                accuracies.append(sum(classifications == id_range) / batch_size)

                # return batch to CPU
                batch_queries = batch_queries.to(torch.device('cpu'))
                batch_targets = batch_targets.to(torch.device('cpu'))
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
            accuracy = sum(classifications == torch.Tensor([ i for i in range(num_queries) ])) / num_queries
            return accuracy




