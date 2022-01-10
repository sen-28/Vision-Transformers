# Vision-Transformers
Implementation of Vanilla Transformer for image classification tasks on the ImageNet and Cassava datasets.

## Aim 
- Applying a pure transformer directly to image classification tasks.
- Showing that reliance on CNNs is not necessary.
- Proving that Vision Transformer (ViT) attains better results compared to SOTA convolutional networks while requiring substantially fewer computational resources.
 

## Overview of the Model

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/transformers-whole.png)

## Methodolgy

- An image is split into a sequence of fixed-size non-overlapping patches.
- Flattened patches are linearly embedded.
- A [CLS] token is added to serve as representation of entire image. This learnable token is used for classification at output.
- Absolute position embeddings are added.
- Resultant Vector is fed into the Transformer Encoder. 
- The Transformer encoder consists of alternating layers of Multi-headed Self Attention (MSA) and MLP blocks. Layernorm (LN) is applied before every block.

## Splitting image into patches

**x ∈ R <sup>H×W×C</sup>  →  x<sub>p</sub> ∈ R<sup> N×(P.P.C)</sup>**

(H, W) is the resolution of the original image. <br/>
C is the number of channels. <br/>
(P, P) is the resolution of each image patch. <br/>
N = HW/P<sup>2</sup> is the resulting number of patches, effective input sequence length. <br/>

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/128160932-6c92920e-b996-4208-9f71-c5caeb4d7285.png)

## Patch and Positional Embeddings

The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection.

z<sub>0</sub> = [x<sub>class</sub>; x<sup>1</sup><sub>p</sub>E; x<sup>2</sup><sub>p</sub>E; · · · ;  x<sup>N</sup><sub>p</sub>E] + E<sub>pos</sub>

E ∈ R<sup> (P .P ·C)×D </sup>  is the patch embedding,  
E<sub>pos</sub> ∈ R <sup>(N+1)×D</sup>  is the positional embedding,
X<sub>class</sub> is the learnable embedding. 

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/pos.png)

## Structure of Encoder

We prepend a learnable embedding to the sequence of embedded patches (z<sup>0</sup><sub>0</sub> = x<sub>class</sub>), whose state at the output of the Transformer encoder (z<sup>0</sup><sub>L</sub>) serves as the image representation y. <br/>

y = LN(z<sup>0</sup><sub>L</sub>) <br/>
LN is LayerNorm, <br/>
L is the total number of layers in the Transformer encoder. <br/>

Both during pre-training and fine-tuning, a classification head is attached to z<sup>0</sup><sub>L</sub>. <br/>

**The Transformer encoder consists of alternating layers of multi-headed self-attention (MSA) and multi-layer perceptron (MLP) blocks. LayerNorm (LN) is applied before every block, and residual connections after every block.** <<br/>

The MLP contains two layers with a GELU non-linearity. 


## ImageNet - Big Dataset

**We used the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images.** 

ImageNet is an image database organized containing more than 14 million  images have been hand-annotated by the project to indicate what objects are pictured and in at least one million of the images, bounding boxes are also provided.

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/imagenet.png)


## Cassava - Small Dataset (Fine-tuning)

**We used the Cassava dataset consisting of a total of 21367 labelled images.**

Cassava consists of leaf images for the cassava plant depicting healthy and four (4) disease conditions. 

- Cassava Mosaic Disease(CMD)
- Cassava Bacterial Blight (CBB) 
- Cassava Green Mite (CGM) 
- Cassava Brown Streak Disease (CBSD)
- Healthy

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/cas.png)


## Preprocessing

**Pytorch was used for the code and training time was 55 minutes on Colab GPU.**

- Train dataset - Normalization
- Random transformations - horizontal and vertical flip
- Resizing to 224*224
- Validation dataset - Resizing to 224*224

## Model architecture

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/transform.png)

- **Patch size** is specified as **16** and **196** patches of size **16x16** are obtained from the image.
- **Number of channels** are **3**. Convolutional 2d layer is used for reshaping the input image as a sequence of flattened patches. - **Kernel** and **stride size** is **16**. We map to **768 dimensions**. 
- We add a **learnable embedding** to this sequence whose state at output gives us the class.
- **Position embeddings** are added to retain positional information.
- Each block of the transformer has **Layer Normalization**. This is followed by an attention layer with 8 heads. The input is also fed into **MLP layer**. Outputs from MLP and attention layer are added. 
- The MLP layer has a **linear layer** followed by **GELU activation**. This is again followed by a **linear layer**.
- We use **12 blocks** for our implementation. There are **residual connections** after every block.

## Self-Attention 

Attention in layman terms is the process of selectively concentrating on a few things while ignoring other aspects. Takes in an **input of size n** and gives out an **output of size n**. Uses **Key, Query** and **Value** matrices. 

**Self-Attention (SA):** For each element in an input sequence z ∈ R <sup> N X D </sup> we compute a weighted sum over all values v in the sequence. The attention weights A<sub>ij</sub> are based on the pairwise similarity between two elements of the sequence and their respective query q<sup>i</sup> and key k<sup>j</sup> representations.

**Multihead self-attention (MSA) is an extension of SA in which we run k self-attention operations, called “heads”, in parallel, and project their concatenated outputs.**

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/attnetion.jpeg)

## Implementation details 

- Training time - 55 minutes on Colab GPU
- Framework used - PyTorch
- Number of epochs - 10
- Learning rate - 0.001 with scheduling
- Optimizer - SGD
- Loss function - Cross-entropy loss
- Batch size - 16
- Training accuracy after 10 epochs - 86.888
- Validation accuracy after 10 epochs - 82.00


## Results

**Using the ViT model, we have explored the direct application of Transformers to image recognition.**

Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step. Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP.

This simple, yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets.

**Thus, Vision Transformer matches or exceeds the state of the art on many image classification datasets, whilst being relatively cheap to pre-train.**

## Results - Loss

![alt text](https://github.com/sen-28/Vision-Transformers/blob/main/vision_images/results.png)

Due to GPU contraints, the following steps were carried out for the datasets. This also had an impact on the results obtained.

**ImageNet for Pre-Training -** It was not practically possible to train the ImageNet dataset on Vision transformer with a single Colab GPU. I defined the architecture and loaded pre-trained weights from the official Github repository.

**Cassava Leaf Disease Classification -** I fine-tuned on a subset of 1500 images. I used 10% of these for cross-validation. I also reduced the number of classes to 3 instead of 5. 

## Future scope

The follow-up works that have been made on Vision transformers are very encouraging. They include - 

- **Applying ViT to other computer vision tasks, such as detection and segmentation.** DINO (a method for self-supervised training of Vision Transformers) by Facebook AI. DINO models are capable of segmenting objects, without having ever been trained to do so. 
- **Further scaling of ViT would likely lead to improved performance.** ViT models haven’t shown any signs of saturating performance yet.
- **Zero-shot learning**














