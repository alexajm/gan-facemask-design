# Design of Facemasks to Protect from Facial Recognition using Generative Adversarial Networks

## Summary

Deep learning has empowered substantial improvements in facial identification technology. While some applications are benign, such as smart phones that unlock when they recognize their user’s face, many of the *realized* applications of facial identification have dangerous implications. For example, while many promote the use of facial id to catch "criminals," in reality it gives the state tremendous power to target activists, minority populations, the homeless, and other members of our communities. As such, this project seeks to develop deep learning tools to protect individuals from facial identification in scenarios when they do not consent to it. One of the most cost-effective tools against facial recognition is a facemask.  However, recent research has sought to modify facial recognition algorithms to identify people behind their masks. In this project, I propose a network for generating facemask designs that elude recognition. In particular, I use a generative adversarial network (GAN) to generate and evaluate designs.

![network](/figures/network.png)

High-level network architecture is as follows: (a) generator creates mask design; (b) mask is projected onto training data; (c) masked images are passed to discriminator; (d) discriminator compares masked images against pre-computed embeddings of unmasked images for facial recognition; (e) loss from classification task is used to update generator and discriminator parameters.