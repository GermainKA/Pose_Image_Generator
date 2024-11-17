# Dance Video Generation

## Introduction

The goal of this project is to generate a new video of a target person performing the same movements as a source person. Given two videos—one of a source person and another of a target person—we aim to transfer the motion from the source to the target, creating a synthesized video where the target imitates the source's movements.

## Methods Implemented

We have implemented four methods to achieve this goal:

### 1. GenNearest

**Description**: This method finds the closest matching frame in the target video for each frame in the source video based on skeletal data. It computes the distance between the skeleton of the source frame and all skeletons in the target video, selecting the frame with the minimum distance.

**Key Points**:
- Uses nearest neighbor search based on skeletal joint coordinates.
- Simple and fast implementation.
- May produce jittery results due to frame-by-frame matching without temporal consistency.

### 2. GenVanillaNN (Skeleton Coordinates as Input)

**Description**: This method uses a simple neural network (Vanilla Neural Network) that takes the skeleton coordinates as input and generates the corresponding image of the target person in that pose.

**Key Points**:
- **Input**: Skeleton joint coordinates.
- **Output**: Synthesized image of the target person.
- The network learns to map skeletal positions to images.
- Requires training on paired data of skeletons and images from the target video.

### 3. GenVanillaNN (Skeleton Image as Input)

**Description**: Similar to the previous method, but instead of using skeleton coordinates, it uses images of the skeleton (visual representations) as input to the neural network.

**Key Points**:
- **Input**: Images of skeletons.
- **Output**: Synthesized image of the target person.
- May capture spatial relationships better due to image input.
- Also requires training on paired data.

### 4. GenGAN

**Description**: This method employs a Generative Adversarial Network (GAN) to generate images of the target person in the poses specified by the source skeletons.

**Key Points**:
- Consists of a generator and a discriminator network.
- The generator aims to produce realistic images of the target person in new poses.
- The discriminator tries to distinguish between real images from the target video and the generated images.
- Adversarial training helps in producing higher quality and more realistic images.

### Prerequisites

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Other dependencies as required by the code

## Using the Start Code
We used the start code provided at the following address:
[Start code](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/doc/tp_dance_start.zip)