# Vision-Transformers
Implementation of Vanilla Transformers for image classification on the ImageNet and Cassava datasets

**Aim** 
- Applying a pure transformer directly to image classification tasks.
- Showing that reliance on CNNs is not necessary. 
- Proving that Vision Transformer (ViT) attains better results compared to SOTA convolutional networks while requiring substantially fewer computational resources.

**Methodology**

- An image is split into a sequence of fixed-size non-overlapping patches.
- Flattened patches are linearly embedded.
- A [CLS] token is added to serve as representation of entire image. This learnable token is used for classification at output.
- Absolute position embeddings are added.
- Resultant Vector is fed into the Transformer Encoder. 
- The Transformer encoder consists of alternating layers of Multi-headed Self Attention (MSA) and MLP blocks. Layernorm (LN) is applied before every block.







