# Design of Facemasks to Protect from Facial Recognition using Generative Adversarial Networks

## Summary

Deep learning has empowered substantial improvements in facial recognition technology. While some applications of facial recognition are benign, many have dangerous implications. For example, facial recognition gives both the state and malicious actors tremendous power to target minority populations, activists, and other members of our communities. This project therefore seeks to develop deep learning tools to project individuals from facial recognition in scenarios where they do not consent to it. In particular, this paper presents a specialized Generative Adversarial Network (GAN) for designing facemasks that elude recognition.

![network](/figures/network.png)

High-level network architecture is as follows: (a) generator creates mask design; (b) mask is projected onto training data; (c) masked images are passed to discriminator; (d) discriminator compares masked images against pre-computed embeddings of unmasked images for facial recognition; (e) loss from classification task is used to update generator and discriminator parameters.

## Setupv

This project uses Python 3.7.5 and can be run from a Python virtual environment. Setup is as follows:

1. Download Python 3.7.5

2. Clone the project repo

```
git clone https://github.com/alexajm/gan-facemask-design.git
```

3. Navigate to the project directory

```
cd gan-facemask-design
```

4. Create and activate a virtual environment

```
python3.7 -m venv venv
source venv/bin/activate
```

5. Download requirements

```
pip install -r requirements.txt
```

6. ...

7. Deactivate the virtual environment when you're done

```
deactivate
```