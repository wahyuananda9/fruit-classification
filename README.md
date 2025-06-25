# Fruit Image Classification with CNN

This repository contains the code and resources to build, train, and deploy a Convolutional Neural Network (CNN) model to classify 90 different types of fruits. The model was trained on the "Fruits 360" dataset and achieved very high accuracy.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyyou50/fruits_classification)

## Table of Contents
1.  [Project Description](#project-description)
2.  [Dataset](#dataset)
3.  [Model Architecture](#model-architecture)
4.  [Results](#results)
5.  [Installation](#installation)
6.  [How to Use](#how-to-use)
7.  [Deployment](#deployment)
8.  [File Structure](#file-structure)

## Project Description
This project implements a CNN using TensorFlow and Keras for an image classification task. The goal is to create a model that can accurately identify the type of fruit from an image. The project covers all steps from data preprocessing, model building, training, evaluation, to deployment on a public platform.

## Dataset
* **Dataset Name**: Fruits 360.
* **Source**: Downloaded via `kagglehub` from the `moltean/fruits` dataset.
* **Specifications**:
    * **Number of Classes**: 90.
    * **Training Data**: 26,335 images.
    * **Validation Data**: 2,887 images.
    * **Test Data**: 14,527 images.
    * **Input Resolution**: Images are resized to `500x500` pixels.

## Model Architecture
The model uses Keras's `Sequential` API. The main layers are as follows:
* `Conv2D(32, (3,3), activation='relu', input_shape=(500, 500, 3))`
* `MaxPooling2D(2, 2)`
* `Conv2D(64, (3,3), activation='relu')`
* `MaxPooling2D(2,2)`
* `Flatten()`
* `Dense(128, activation='relu')`
* `Dropout(0.5)`
* `Dense(90, activation='softmax')`

The model was compiled with the `adam` optimizer and `categorical_crossentropy` loss function.

## Results
After training for 30 epochs, the model was evaluated on the test set and achieved a highly satisfactory result.
* **Test Accuracy**: **99.62%**.

## Installation
To run this project locally, ensure you have Python and `pip` installed. Then, install the required libraries:
```bash
pip install tensorflow keras pillow scikit-learn matplotlib
