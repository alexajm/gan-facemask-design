# Design of Facemasks to Protect from Facial Recognition using Generative Adversarial Networks

## Summary

Deep learning has empowered substantial improvements in facial recognition technology. While some applications of facial recognition are benign, many have dangerous implications. For example, facial recognition gives both the state and malicious actors tremendous power to target minority populations, activists, and other members of our communities. This project therefore seeks to develop deep learning tools to project individuals from facial recognition in scenarios where they do not consent to it. In particular, this paper presents a specialized Generative Adversarial Network (GAN) for designing facemasks that elude recognition.

![network](/figures/network.png)

High-level network architecture is as follows: (a) generator creates mask design; (b) mask is projected onto training data; (c) masked images are passed to discriminator; (d) discriminator compares masked images against pre-computed embeddings of unmasked images for facial recognition; (e) loss from classification task is used to update generator and discriminator parameters.

## Setup

This project uses Python 3.7.5 and can be run from a Python virtual environment. Setup is as follows:

1. Download Python 3.7.5

2. Clone the project repo

```
% git clone https://github.com/alexajm/gan-facemask-design.git
```

3. Navigate to the project directory

```
% cd gan-facemask-design
```

4. Create and activate a virtual environment

```
% python3.7 -m venv venv
% source venv/bin/activate
```

5. Download requirements

```
% pip install -r requirements.txt
```

6. When you're done working with the GAN, deactivate the virtual environment

```
% deactivate
```

## Use

The code for training the GAN can be found in two different locations: `main.py` and `sandbox.ipynb`.

`main.py` is optimized for CUDA and can be run from a command line argument, making it ideal for training the network on remote machines (e.g. via AWS). Use `python main.py -h` to see argument options.

`sandbox.ipynb` is similar to `main.py`, but organized as a Jupyter notebook for prototyping and debugging the models. It offers quick visuals for many of the individual project components, such as the datasets and GAN outputs. Use `jupyter notebook sandbox.ipynb` to open the sandbox.

## Organization

Project files are organized into four separate directories:

* `code` - project code
* `data` - test and training data
* `figures` - final and intermediate model results
* `models` - saved `pytorch` models for the generator and projector
