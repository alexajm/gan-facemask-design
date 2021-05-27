import torch


def shuffle_data(inputs, outputs):
    '''Shuffle the first dimension of a set of input/output data'''
    n_examples = outputs.shape[0]
    shuffled_indices = torch.randperm(n_examples)
    inputs = inputs[shuffled_indices]
    outputs = outputs[shuffled_indices]
    return inputs, outputs


def batch_data(inputs, outputs, batch_size=32):
    '''Convert full input/output pairs to a list of batched tuples'''
    n_examples = outputs.shape[0]
    return [ (inputs[batch_size * i:batch_size * (i+1)],
              outputs[batch_size * i:batch_size * (i+1)])
             for i in range(n_examples // batch_size) ]


def train_batch(model, batch):
    '''Perform one iteration of model training given a single batch'''
    inputs, correct_outputs = batch
    model_outputs = model.forward(inputs)
    model.optimizer.zero_grad()
    loss = model.criterion(model_outputs, correct_outputs)
    loss.backward(retain_graph=True)
    model.optimizer.step()
    return float(loss) / model_outputs.shape[0]


def fit(model, inputs, correct_outputs, num_epochs=10):
    ''' Train model on input/output pairs over desired number of epochs'''
    epoch_loss = 0
    for epoch in range(1, num_epochs + 1):
        # sort data into minibatches
        inputs, correct_outputs = shuffle_data(inputs, correct_outputs)
        minibatches = batch_data(inputs, correct_outputs)

        # train on each minibatch
        epoch_loss = 0
        for batch in minibatches:
            epoch_loss += train_batch(model, batch)
        epoch_loss /= len(minibatches)

        # output loss
        if epoch % 10 == 0: print('Epoch {} loss: {}'.format(epoch, epoch_loss))

    # return training accuracy
    return epoch_loss
