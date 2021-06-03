# imports
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pdb
import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
import styles
import sys
from gan import Generator, Projector, Discriminator, GAN
import argparse


def main(num_faces, cuda, verbose=False):
    # set computation device (None/CPU if in development mode, CUDA otherwise)
    device = torch.device('cuda:0') if cuda else None

    # load masked and unmasked faces
    img_size = 128
    masked_faces = torch.zeros((num_faces, 3, 128, 128))
    unmasked_faces = torch.zeros((num_faces, 3, 128, 128))
    face_id = 0
    skipped = 0
    idx_to_face_id = []
    while face_id < num_faces:
        # get image paths
        face_id_text = str(100000 + face_id + skipped)[1:]
        masked_path = '../data/masked/{}_Mask.jpg'.format(face_id_text)
        unmasked_path = '../data/unmasked/{}.png'.format(face_id_text)
        
        # load images
        try:
            masked_img = Image.open(masked_path)
            unmasked_img = Image.open(unmasked_path)
        except:  # skip missing images
            skipped += 1
            continue
            
        # process images
        masked_tensor = Resize((img_size,img_size))(ToTensor()(masked_img)) - .5
        unmasked_tensor = Resize((img_size,img_size))(ToTensor()(unmasked_img)) - .5
        if masked_tensor is None or unmasked_tensor is None:
            skipped += 1
            continue
        
        # save images to respective tensors
        masked_faces[face_id] = masked_tensor
        unmasked_faces[face_id] = unmasked_tensor
        idx_to_face_id.append(int(face_id + skipped))
        face_id += 1

    # instantiate GAN
    generator = Generator(learning_rate=2e-3)
    projector = Projector(load_path='../models/projector.pt')
    discriminator = Discriminator()
    gan = GAN(generator, projector, discriminator, device=device)

    # compute and store unmasked discriminator embeddings
    gan.compute_unmasked_embeddings(unmasked_faces)

    # split data into training and testing sets
    split = int(.8 * num_faces)
    train_input, train_output = masked_faces[:split], torch.Tensor(idx_to_face_id[:split]).long()
    test_input, test_output = masked_faces[split:], torch.Tensor(idx_to_face_id[split:]).long()

    # train
    gan.fit(train_input, train_output, num_epochs=1)
    # gan.fit(train_input, train_output, num_epochs=100)
    # gan.learning_rate = 2e-4
    #gan.fit(train_input, train_output, num_epochs=100)

    # save generator model
    save_path = '../models/generator.pt'
    torch.save(gan.generator.state_dict(), save_path)

    # display sample masks and faces
    plt.figure()
    fig, axes = plt.subplots(2, 5)
    fig.set_figwidth(20)
    fig.set_figheight(7)
    for idx in range(5):
        # original image
        face_id = idx_to_face_id[idx]
        face_id_text = str(100000 + face_id)[1:]
        original_img = Image.open('../data/masked/{}_Mask.jpg'.format(face_id_text))
        axes[0,idx].imshow(original_img)
        axes[0,idx].get_xaxis().set_ticks([])
        axes[0,idx].get_yaxis().set_ticks([])
        
        # generated mask image
        mask = gan.generator()
        masked_tensor = masked_faces[idx].unsqueeze(0)
        if device: masked_tensor.to(device)
        masked_image = gan.project_mask(mask, masked_tensor, process=True)[0]
        masked_image = torch.transpose(masked_image, 0, 1)
        masked_image = torch.transpose(masked_image, 1, 2)
        masked_image = masked_image.cpu().detach().numpy()
        axes[1,idx].imshow(masked_image)
        axes[1,idx].get_xaxis().set_ticks([])
        axes[1,idx].get_yaxis().set_ticks([])
    plt.savefig('../figures/sample_masks.png')

    # evaluate accuracy
    train_accuracy = gan.evaluate(train_input, train_output)
    test_accuracy = gan.evaluate(test_input, test_output)
    masked_accuracy = gan.discriminator.evaluate(masked_faces, unmasked_faces)
    unmasked_accuracy = gan.discriminator.evaluate(unmasked_faces, unmasked_faces)
    print('facial recognition accuracy for...')
    print('   random choice:\t\t{:.1f}%'.format(100 / num_faces))
    print('   training images:\t\t{:.1f}%'.format(100 * train_accuracy))
    print('   testing images:\t\t{:.1f}%'.format(100 * test_accuracy))
    print('   original masked images:\t{:.1f}%'.format(100 * masked_accuracy))
    print('   original unmasked images:\t{:.1f}%'.format(100 * unmasked_accuracy))

    # write results to file
    file_path = '../data/accuracy.txt'
    with open(file_path, 'w') as file:
        file.write('facial recognition accuracy for...')
        file.write('\n   random choice:\t\t{:.1f}%'.format(100 / num_faces))
        file.write('\n   training images:\t\t{:.1f}%'.format(100 * train_accuracy))
        file.write('\n   testing images:\t\t{:.1f}%'.format(100 * test_accuracy))
        file.write('\n   original masked images:\t{:.1f}%'.format(100 * masked_accuracy))
        file.write('\n   original unmasked images:\t{:.1f}%'.format(100 * unmasked_accuracy))


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('num_faces', help='number of faces to train the network on', type=int)
    parser.add_argument('--cuda', help='turn on CUDA processing (use GPU instead of CPU)', action='store_true')
    parser.add_argument('-v', '--verbose', help='show verbose outputs', action='store_true')
    args = parser.parse_args()

    # run code
    main(args.num_faces, args.cuda, args.verbose)



