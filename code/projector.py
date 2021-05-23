import numpy as np
import torch
from torch import nn, optim


class Projector(nn.Module):
    def __init__(self, learning_rate=1e-3):
        # initialize
        super(Projector, self).__init__()

        # layers
        # input: 128x128 RGB image
        # output: 128x128 transparency mask
        self.h1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.h2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.h3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=18, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
            )

        # optimizer (Adam)
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

        # loss (MSE)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y = self.h1(x)
        y = self.h2(y)
        y = self.h3(y)
        y = self.output(y)
        return y

    def shuffle_data(self, inputs, outputs):
        n_examples = outputs.shape[0]
        shuffled_indices = torch.randperm(n_examples)
        inputs = inputs[shuffled_indices,:,:,:]
        outputs = outputs[shuffled_indices,:,:,:]
        return inputs, outputs

    def batch_data(self, inputs, outputs, batch_size=16):
        n_examples = outputs.shape[0]
        return [ (inputs[batch_size * i:batch_size * (i+1),:,:,:], outputs[batch_size * i:batch_size * (i+1),:,:]) for i in range(n_examples // batch_size) ]

    def train_batch(self, batch):
        inputs, correct_outputs = batch
        batch_outputs = self.forward(inputs)
        self.optimizer.zero_grad()
        loss = self.criterion(batch_outputs.float(), correct_outputs.float())
        loss.backward()
        self.optimizer.step()
        return float(loss) / len(batch_outputs)

    def fit(self, inputs, outputs, num_epochs=10):
        # set model to train mode
        self.train()

        # train over desired number of epochs
        epoch_loss = 0
        for epoch in range(1, num_epochs + 1):
            # sort data into minibatches
            inputs, outputs = self.shuffle_data(inputs, outputs)
            minibatches = self.batch_data(inputs, outputs)        

            # train on each minibatch
            epoch_loss = 0
            for batch in minibatches:
                epoch_loss += self.train_batch(batch)
            epoch_loss /= len(minibatches)

            # output loss
            if epoch % 10 == 0: print('Epoch {} loss: {}'.format(epoch, epoch_loss))

        # return training accuracy
        return epoch_loss

    def predict(self, x):
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 0)
        y = self.forward(x)
        return y

    def evaluate(self, inputs, correct_outputs):
        # set model to eval mode
        self.eval()

        # forward propagation
        batch_outputs = self.forward(inputs)

        # return test loss
        loss = self.criterion(batch_outputs.float(), correct_outputs.float())
        n_examples = correct_outputs.shape[0]
        return float(loss) / n_examples





