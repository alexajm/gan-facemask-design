import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt


def shuffle_data(inputs, outputs):
    """Shuffle the first dimension of a set of input/output data"""
    n_examples = outputs.shape[0]
    shuffled_indices = torch.randperm(n_examples)
    inputs = inputs[shuffled_indices]
    outputs = outputs[shuffled_indices]
    return inputs, outputs


def batch_data(inputs, outputs, batch_size=16):
    """Convert full input/output pairs to a list of batched tuples"""
    n_examples = outputs.shape[0]
    return [
        (
            inputs[batch_size * i : batch_size * (i + 1)],
            outputs[batch_size * i : batch_size * (i + 1)],
        )
        for i in range(n_examples // batch_size)
    ]


def train_batch(model, batch):
    """Perform one iteration of model training given a single batch"""
    # send data to CUDA if necessary
    inputs, correct_outputs = batch
    if model.device:
        inputs = inputs.to(model.device)
        correct_outputs = correct_outputs.to(model.device)

    # train batch
    model_outputs = model.forward(inputs)
    model.optimizer.zero_grad()
    loss = model.criterion(model_outputs, correct_outputs)
    loss.backward(retain_graph=True)
    model.optimizer.step()

    # return data to CPU if necessary
    if model.device:
        inputs = inputs.to(torch.device("cpu"))
        correct_outputs = correct_outputs.to(torch.device("cpu"))
    return float(loss) / model_outputs.shape[0]


def fit(model, inputs, correct_outputs, num_epochs=10):
    """Train model on input/output pairs over desired number of epochs"""
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
        if epoch % 10 == 0:
            print("Epoch {} loss: {}".format(epoch, epoch_loss))

    # return training accuracy
    return epoch_loss


def load_faces(num_faces, dir_1, suffix_1, dir_2, suffix_2):
    # confirm that directories exist
    assert os.path.isdir(dir_1)
    assert os.path.isdir(dir_2)

    # load data
    img_size = 128
    faces_1 = torch.zeros((num_faces, 3, img_size, img_size))
    faces_2 = torch.zeros((num_faces, 3, img_size, img_size))
    plt.figure()
    fig, axes = plt.subplots(2, 5)
    fig.set_figwidth(20)
    fig.set_figheight(7)
    face_id = 0
    skipped = 0
    idx_to_face_id = []
    while face_id < num_faces and skipped < num_faces:
        # get image paths
        face_id_text = str(100000 + face_id + skipped)[1:]
        path_1 = "{}/{}{}".format(dir_1, face_id_text, suffix_1)
        path_2 = "{}/{}{}".format(dir_2, face_id_text, suffix_2)

        # load images
        try:
            img_1 = Image.open(path_1)
            img_2 = Image.open(path_2)
        except:  # skip missing images
            skipped += 1
            continue

        # process images
        tensor_1 = Resize((img_size, img_size))(ToTensor()(img_1)) - 0.5
        tensor_2 = Resize((img_size, img_size))(ToTensor()(img_2)) - 0.5
        if tensor_1 is None or tensor_2 is None:
            skipped += 1
            continue

        # display images
        if face_id < 5:
            axes[0, face_id].imshow(img_1)
            axes[0, face_id].set_axis_off()
            axes[1, face_id].imshow(img_2)
            axes[1, face_id].set_axis_off()

        # save images to respective tensors
        faces_1[face_id] = tensor_1
        faces_2[face_id] = tensor_2
        idx_to_face_id.append(face_id_text)
        face_id += 1

    # warn that `num_faces` was too large
    if skipped > num_faces:
        print(
            "NOTE: `num_faces` is larger than total capacity of chosen directories ({}, {})".format(
                dir_1, dir_2
            )
        )

    # return
    return faces_1, faces_2, idx_to_face_id
