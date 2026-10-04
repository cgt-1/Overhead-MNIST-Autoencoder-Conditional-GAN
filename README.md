# Overhead MNIST: Autoencoder & Conditional GAN

A deep learning project using the **Overhead MNIST** satellite image dataset to explore two generative deep learning tasks:

1. **Dimensionality reduction** using a convolutional autoencoder
2. **Conditional image generation** using a conditional GAN (cGAN)

The project uses only the **car and plane** classes and compares a baseline model with a modified architecture for each task.

---

## 📌 Project Overview

The project consists of two independent experiments.

### Part 1 — Autoencoder

A convolutional autoencoder compresses **28×28 grayscale images (784 pixels)** into a **128-dimensional latent representation** and reconstructs the original images.

Reconstruction quality is evaluated using **Structural Similarity Index (SSIM)**, where higher values indicate better similarity.

### Part 2 — Conditional GAN

A conditional GAN generates new **car and plane satellite images** based on a class label.

Generation quality is evaluated using **Fréchet Inception Distance (FID)**, where lower values indicate closer similarity between real and generated image distributions.

Both parts compare a baseline model against a modified version to examine the effect of increased model capacity and different training strategies.

---

## 📂 Dataset

The project uses the **[Overhead MNIST dataset](https://www.kaggle.com/datasets/datamunge/overheadmnist/data)**, specifically the `version2` folder.

The original dataset contains 10 classes; this project uses:

| Class     | Train | Test |      Total |
| --------- | ----: | ---: | ---------: |
| Car       | 7,116 |  896 |      8,012 |
| Plane     | 7,132 |  888 |      8,020 |
| **Total** |       |      | **16,032** |

The dataset is **not included in the GitHub repository due to its size**.

📁 **Dataset:** [Google Drive — Overhead MNIST Dataset](YOUR-GOOGLE-DRIVE-LINK-HERE)

The notebook mounts the Google Drive dataset directly when running on Google Colab.

---

## 🧹 Data Preparation

The original train and test folders were combined and reshuffled before creating a new split:

* **80% training:** 12,825 images
* **10% validation:** 1,603 images
* **10% test:** 1,604 images
* Images converted to grayscale and resized to **28×28**
* Both car and plane classes are used together

Different pixel scaling was used for each model:

* **Autoencoder:** `[0, 1]`
* **cGAN:** `[-1, 1]` to match the generator's `tanh` output

---

## 🔹 Part 1: Convolutional Autoencoder

### Architecture

Both models use the same **128-dimensional bottleneck**, allowing the comparison to focus on architecture and training improvements.

|            | Baseline                                     | Modified                                                    |
| ---------- | -------------------------------------------- | ----------------------------------------------------------- |
| Encoder    | Conv2D(32) → MaxPool → Dense(128)            | Conv2D(32) → Conv2D(64) → MaxPool → Dense(256) → Dense(128) |
| Decoder    | Dense → Reshape → Upsampling → Conv2D layers | Larger dense and convolutional layers                       |
| Parameters | ~1.62M                                       | ~6.6M                                                       |
| Optimizer  | Adam, LR 0.001                               | Adam, LR 0.0005                                             |
| Batch Size | 64                                           | 32                                                          |
| Training   | 40 epochs                                    | Up to 100 epochs                                            |
| Loss       | Binary Cross-Entropy                         | Binary Cross-Entropy                                        |

The modified model also uses **EarlyStopping** and **ReduceLROnPlateau**.

### Results

| Model        | Average SSIM |
| ------------ | -----------: |
| Baseline     |       0.7648 |
| **Modified** |   **0.8260** |

The modified autoencoder improved SSIM by approximately **0.06**, or around **8%**, demonstrating noticeably better reconstruction quality.

---

## 🔹 Part 2: Conditional GAN

The cGAN consists of a **generator** and **discriminator** that are both conditioned on the image class.

The generator receives random noise and a class label and produces either a car or plane image. The discriminator receives an image and its class label and predicts whether the image is real or generated.

### Training Comparison

|                  | Baseline                | Modified                |
| ---------------- | ----------------------- | ----------------------- |
| Generator        | 128 → 256 → 512 → 1024  | Same                    |
| Discriminator    | 512 → 1024 → 1024 → 512 | Same                    |
| Generator LR     | 0.0002                  | 0.0002                  |
| Discriminator LR | 0.0002                  | **0.0001**              |
| Label smoothing  | None                    | **0.9 for real images** |
| Epochs           | 50                      | 50                      |
| Batch Size       | 64                      | 64                      |

The modified model keeps the same architecture but uses a **slower discriminator** and **one-sided label smoothing** to prevent the discriminator from overpowering the generator.

### FID Results

| Model             |       FID |
| ----------------- | --------: |
| Baseline cGAN     |     27.73 |
| **Modified cGAN** | **24.90** |

**Lower FID is better.** In this run, the modified cGAN achieved approximately **10% lower FID** than the baseline.

> ⚠️ **FID Result Disclaimer:** GAN training is stochastic, so FID can vary between runs. PyTorch random seeds were not fixed, meaning weight initialization, training order, and generated noise can differ each time. For example, an earlier baseline run produced an FID of **21.72**, while the final baseline run produced **27.73**. Since this variation is larger than the 2.83-point difference between the baseline and modified models, the FID comparison should be considered **indicative rather than conclusive**. The reported values represent a single run for each model.

Also, FID in this project is calculated using **flattened 784-dimensional pixel vectors** rather than standard Inception features because the images are small 28×28 grayscale images. Therefore, these FID values **should not be directly compared with FID scores from other studies using standard Inception-based FID**.

### Key Findings

* The modified cGAN achieved a lower FID in the reported run.
* GAN training remained unstable, with the discriminator continuing to improve while the generator's loss increased after approximately 18–20 epochs.
* Generated images captured general brightness and contrast patterns but had difficulty reproducing fine-grained object shapes.
* Saving checkpoints and evaluating FID throughout training could help identify better generator weights.

---

## 🧠 Main Takeaways

This project demonstrates several practical deep learning concepts:

* Convolutional autoencoders for dimensionality reduction
* Latent-space representation learning
* Conditional image generation
* GAN generator/discriminator dynamics
* SSIM-based image reconstruction evaluation
* FID-based generative model evaluation
* Early stopping and learning-rate scheduling
* The importance of accounting for randomness when evaluating GANs

An important result from the experiments is that **larger or more complex models do not automatically produce better results**. The modified autoencoder improved substantially, while the modified cGAN showed only a small FID improvement that may not be statistically meaningful due to run-to-run variation.

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras** — convolutional autoencoder
* **PyTorch / torchvision** — conditional GAN
* **scikit-image** — SSIM
* **SciPy & NumPy** — FID calculation
* **scikit-learn** — dataset splitting
* **OpenCV, pandas & Matplotlib** — image processing and visualization
* **Google Colab + GPU** — model training

---

## 📁 Project Structure

```text
├── MNISTAutoencoderGAN.ipynb
└── README.md
```

The dataset is stored separately in Google Drive rather than committed to the repository due to its size.

---

## 🔮 Future Improvements

### Autoencoder

* Experiment with different latent-space sizes such as 32, 64, and 128.
* Explore convolutional bottlenecks or variational autoencoders.
* Visualize latent representations using PCA or t-SNE.

### cGAN

* Use convolutional DCGAN-style architectures.
* Reduce discriminator capacity to improve generator/discriminator balance.
* Save checkpoints and evaluate FID throughout training.
* Run multiple seeds and report mean ± standard deviation for FID.
* Explore standard Inception-based FID with an appropriate image preprocessing strategy.

---

## 📚 References

* [Overhead MNIST — Kaggle](https://www.kaggle.com/datasets/datamunge/overheadmnist/data)
* [Overhead MNIST Paper — arXiv](https://arxiv.org/pdf/2102.04266)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [PyTorch Documentation](https://pytorch.org/)
