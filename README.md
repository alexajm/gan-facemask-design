# Design of Facemasks to Protect from Facial Identification using Generative Adversarial Networks

## Summary

Deep learning has empowered substantial improvements in facial identification technology. While some applications are benign, such as smart phones that unlock when they recognize their user’s face, many of the \textit{realized} applications of facial identification have dangerous implications. For example, while many promote the use of facial id to catch "criminals," in reality it gives the state tremendous power to target activists, minority populations, the homeless, and other members of our communities. As such, this project takes the standpoint that facial identification can be a dangerous technology, and seeks to develop deep learning tools to protect from it. One of the most cost-effective tools against facial recognition is a facemask.  However, recent research has sought to modify facial recognition algorithms to identify people behind their masks. In this project, I propose a network for generating facemask designs that elude recognition. In particular, I use a generative adversarial network (GAN) to generate and evaluate designs.

![alt text](https://github.com/alexajm/gan-facemask-design/figures/network.png?raw=true)

Draft network architecture. (a) Generator creates mask design. (b) Mask design is pasted onto training data. (c) Masked images are passed to discriminator for facial recognition. (d) Loss from classification task is used to update generator and discriminator parameters.