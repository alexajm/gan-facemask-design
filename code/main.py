# imports
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pdb
import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
import sys
from gan import Generator, Projector, Discriminator, GAN
import argparse
import time
import os
import utils


torch.autograd.set_detect_anomaly(True)


def main(num_faces, num_epochs, cuda, verbose=False):
    # set computation device (None/CPU if in development mode, CUDA otherwise)
    device = torch.device("cuda:0") if cuda else None

    # load faces
    masked_dir = "../data/masked"
    masked_suffix = "_Mask.jpg"
    unmasked_dir = "../data/unmasked"
    unmasked_suffix = ".png"
    masked_faces, unmasked_faces, idx_to_face_id = utils.load_faces(
        num_faces, masked_dir, masked_suffix, unmasked_dir, unmasked_suffix
    )
    if verbose:
        print("loaded {} faces...".format(num_faces))

    # split data into training and testing sets
    split = int(0.8 * num_faces)
    train_input, train_output = (
        masked_faces[:split],
        torch.Tensor((range(0, split))).long(),
    )
    test_input, test_output = (
        masked_faces[split:],
        torch.Tensor(range(split, num_faces)).long(),
    )
    static_faces = unmasked_faces[:split]

    # instantiate GAN
    generator = Generator(learning_rate=2e-3)
    projector = Projector(load_path="../models/projector.pt")
    discriminator = Discriminator()
    gan = GAN(generator, projector, discriminator, device=device)
    if verbose:
        print("instantiated GAN...")

    # compute and store unmasked discriminator embeddings
    gan.compute_unmasked_embeddings(unmasked_faces)

    # train
    if verbose:
        print("training initiated...")
    gan.fit(
        train_input, static_faces, train_output, num_epochs=num_epochs, verbose=verbose
    )
    if verbose:
        print("\ntraining complete...")

    # save models
    save_dir = "../models"
    suffix = time.strftime("%Y%m%d_%H%M%S")
    gan.save(save_dir, suffix)
    if verbose:
        print("models saved under '{}/<model>_{}'...".format(save_dir, suffix))

    # display sample masks and faces
    plt.figure()
    fig, axes = plt.subplots(2, 5)
    fig.set_figwidth(20)
    fig.set_figheight(7)
    for idx in range(5):
        # original image
        face_id = idx_to_face_id[idx]
        original_img = Image.open("../data/masked/{}_Mask.jpg".format(face_id))
        axes[0, idx].imshow(original_img)
        axes[0, idx].get_xaxis().set_ticks([])
        axes[0, idx].get_yaxis().set_ticks([])

        # generated mask image
        mask = (
            gan.generator(torch.rand(1, 100).to(device)) if device else gan.generator()
        )
        masked_tensor = masked_faces[idx].unsqueeze(0)
        if device:
            masked_tensor = masked_tensor.to(device)
        masked_image = gan.project_mask(mask, masked_tensor, process=True)[0]
        masked_image = torch.transpose(masked_image, 0, 1)
        masked_image = torch.transpose(masked_image, 1, 2)
        masked_image = masked_image.cpu().detach().numpy()
        axes[1, idx].imshow(masked_image)
        axes[1, idx].get_xaxis().set_ticks([])
        axes[1, idx].get_yaxis().set_ticks([])
    plt.savefig("../figures/sample_masks.png")

    # evaluate accuracy
    train_accuracy = gan.evaluate(train_input, train_output)
    test_accuracy = gan.evaluate(test_input, test_output)
    masked_accuracy = gan.discriminator_evaluate(masked_faces, unmasked_faces)
    unmasked_accuracy = gan.discriminator_evaluate(unmasked_faces, unmasked_faces)
    print("\nfacial recognition accuracy for...")
    print("   random choice:\t\t{:.1f}%".format(100 / num_faces))
    print("   training images:\t\t{:.1f}%".format(100 * train_accuracy))
    print("   testing images:\t\t{:.1f}%".format(100 * test_accuracy))
    print("   original masked images:\t{:.1f}%".format(100 * masked_accuracy))
    print("   original unmasked images:\t{:.1f}%".format(100 * unmasked_accuracy))

    # write results to file
    file_path = "../data/accuracy.txt"
    with open(file_path, "w") as file:
        file.write("facial recognition accuracy for...")
        file.write("\n   random choice:\t\t{:.1f}%".format(100 / num_faces))
        file.write("\n   training images:\t\t{:.1f}%".format(100 * train_accuracy))
        file.write("\n   testing images:\t\t{:.1f}%".format(100 * test_accuracy))
        file.write(
            "\n   original masked images:\t{:.1f}%".format(100 * masked_accuracy)
        )
        file.write(
            "\n   original unmasked images:\t{:.1f}%".format(100 * unmasked_accuracy)
        )
    if verbose:
        print("\nsaved results...")
        print("done:)")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_faces", help="number of faces to train the network on", type=int
    )
    parser.add_argument(
        "num_epochs", help="number of epochs to train the network over", type=int
    )
    parser.add_argument(
        "--cuda",
        help="turn on CUDA processing (use GPU instead of CPU)",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="show verbose outputs", action="store_true"
    )
    args = parser.parse_args()

    # can't batch <20 faces
    if args.num_faces < 20:
        print("WARNING: cannot batch <20 faces (num_faces = {})".format(args.num_faces))

    # run code
    main(args.num_faces, args.num_epochs, args.cuda, args.verbose)
