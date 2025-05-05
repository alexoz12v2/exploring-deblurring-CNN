# Notes

## Convolutional Autoencoders and Batch Normalization

In a convolutional autoencoder for deblurring, Batch Normalization (BN) can be very helpful for stabilizing training and improving convergence, but using it every time between convolution and ReLU isn't a strict rule—it depends on your architecture and goals. Here’s a practical guide:

- 🔧 Typical Placement: Yes, the most common practice is: Convolution → BatchNorm → ReLU

  - This order works because:

    - Convolution computes the features.
    - BatchNorm normalizes them, reducing internal covariate shift.
    - ReLU adds non-linearity after normalization.

- 🧠 Why Use Batch Normalization in Deblurring?  

  - Stabilizes training when using deeper models.
  - Speeds up convergence by smoothing the optimization landscape.
  - Improves generalization, especially useful in ill-posed tasks like deblurring.

- ⚠️ When You Might Skip BatchNorm:

  - Shallow networks: BN might not be necessary and can introduce overhead.
  - After the final layer: Don’t apply BN right before output (especially if you're reconstructing an image), as it can alter pixel intensity ranges.
  - With small batch sizes: BN may become unstable—consider alternatives like **GroupNorm or InstanceNorm**.

- 🧪 Best Practice:

  - Use BN after each Conv layer and before ReLU, except:
  - Skip BN in the output layer.
  - Be cautious **in the decoder**, especially if your goal is precise pixel-wise reconstruction. In that case, **too much normalization might blur details**.

### GroupNorm and InstanceNorm

Group Normalization (GroupNorm) and Instance Normalization (InstanceNorm) are both alternatives to Batch Normalization (BatchNorm),
especially useful when batch sizes are small or inconsistent.

- **🔹 Instance Normalization (InstanceNorm)**

  - **What it does**: Normalizes each individual sample channel-wise, across spatial dimensions.
    That means it normalizes each channel in each image separately, independent of the batch.
  - **Use case**: Very effective in style transfer and image generation tasks, where individual image features matter more than global statistics across a batch.
  - Works well when batch size is 1 (e.g., inference-time image-to-image tasks).
  - **Formula**: For each sample x in a batch, for each channel:

    ```math
    \text{InstanceNorm}(𝑥) = \frac{𝑥 − 𝜇_{HW}}{𝜎_{HW}}
    ```

    where $𝜇_{HW}$, $𝜎_{HW}$ ​are computed over height and width only (not batch or channels). [Pytorch Link](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html)

- **🔹 Group Normalization (GroupNorm)**

  - **What it does**: Divides channels into groups and normalizes the features within each group, across spatial dimensions and the grouped channels.
  - **Use case**: Designed to overcome batch size dependence. Performs consistently across a wide range of batch sizes. Suitable for object detection,
    segmentation, and image restoration tasks like deblurring.
  - **Formula**: For each group $g$, normalize:

  ```math
  \text{GroupNorm}(x) = \frac{x - \mu_g}{\sigma_g}
  ```

    where $𝜇_𝑔$, $𝜎_𝑔$ ​are computed over the group’s channels and spatial dimensions. [Pytorch Link](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)

#### Comparison: Batch vs Instance vs Group Normalization

- BatchNorm: Normalizes across batch.
- InstanceNorm: Normalizes per-instance, per-channel.
- GroupNorm: Normalizes per-instance, per-group-of-channels.

🆚 Summary Table:

| Norm Type    | Normalizes Across | Sensitive to Batch Size | Good For                    |
| ------------ | ----------------- | ----------------------- | --------------------------- |
| BatchNorm    | Batch + Channel   | ✅ Yes                  | Standard classification     |
| InstanceNorm | H × W             | ❌ No                   | Style transfer, small batch |
| GroupNorm    | Groups × H × W    | ❌ No                   | Detection, deblurring       |

### Decoder and Batch/Group/Instance Normalization

Even though the model's goal is to deblur images, ironically, the decoder itself can introduce blur due to several architectural choices:

**🔹 1. Upsampling Artifacts**: When the decoder reconstructs the image (upsamples), it can use:

- **Transpose Convolutions (a.k.a. Deconvolutions)** → Can produce checkerboard artifacts or blurry edges if kernel/stride combinations aren't carefully chosen.
- **Nearest-neighbor or bilinear upsampling + Conv** → Smoother but can still produce soft (blurry) features if not followed by sharp, learned convolutions.

Fixes:

- Prefer upsampling + convolution (e.g., bilinear → Conv) over plain transposed convolutions. [Pytorch Link](https://pytorch.org/docs/stable/generated/torch.nn.UpsamplingBilinear2d.html#torch.nn.UpsamplingBilinear2d)
- Use stride-1 convolutions after upsampling to sharpen features.
- Try sub-pixel convolution ([PixelShuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)) for cleaner upsampling.

**🔹 2. Overuse of Normalization**: In the decoder, especially the late layers, using normalization layers like BatchNorm or GroupNorm can suppress critical pixel-level details by:

- Smoothing out feature activations.
- Inhibiting the decoder from expressing sharp, localized patterns.

Fixes:

- Avoid or minimize normalization layers in the last few decoder blocks.
- Some models omit normalization entirely in the decoder for image-to-image tasks.

**🔹 3. Loss Function Choices**: If your loss function is overly focused on pixel-wise accuracy (like L2/MSE loss), it tends to favor blurry but "safe" outputs because:

- Sharp details risk higher error if the prediction is even slightly misaligned.

Fixes:

- Add perceptual loss (using VGG features).
- Add adversarial loss (GAN-style training).
- Use edge-preserving losses (e.g., Total Variation loss or gradient loss).

**🔹 4. Bottleneck Information Loss**: If your encoder compresses too much information, the decoder has limited data to reconstruct fine details, leading to blurred outputs.

Fixes:

- Use skip connections (like U-Net) to pass spatial details from encoder to decoder.
- Increase bottleneck capacity or depth.

**✅ Best Practices to Reduce Decoder Blurring**:

- Use upsampling + Conv (not TransposeConv alone).
- Limit normalization near the output.
- Use skip connections for fine detail preservation.
- Incorporate perceptual or adversarial losses.
- Tune decoder depth and filter sizes to balance detail and overfitting.

### Batch Normalization and Dropout

Dropout regularization mechanisms can also be applied to convolutional layers ([Pytorch Link](https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html)), but it poses
problems:

- It clashes with Batch Normalization ([Paper Link](https://arxiv.org/pdf/1801.05134))
- Performs worse on convolutional layers than on fully connected layers

There is a proposal solution called [DropCluster](https://arxiv.org/pdf/2002.02997)

## Deblurring in traditional image processing approach

Image deblurring in image processing (non-deep learning) approaches is a classical problem. When it comes to **motion blur** and **defocus blur**,
particularly spatially varying (depth-dependent) blur, **traditional methods aim to model the blur process explicitly and invert it**.

Here’s a structured breakdown of how it’s typically done:

### 🔧 1. Modeling the Blur

✅ **Motion Blur**:

- Caused by object or camera movement during exposure.
- Approximated by a linear blur kernel (PSF: Point Spread Function), often directional and uniform or non-uniform.
- Can be modeled as a convolution with a motion blur kernel, e.g., a line kernel at some angle and length.

✅ **Defocus Blur**:

- Caused by the scene being out of the lens’s focal plane.
- Modeled as a circular (disk-shaped) PSF, often known as a pillbox filter.
- The radius of the disk increases with distance from the focal plane, which introduces depth-varying blur.

### 🔍 2. Blind vs. Non-Blind Deblurring

- **Non-blind deblurring**: The blur kernel (PSF) is known → invert the convolution.
- **Blind deblurring**: PSF is unknown and must be estimated from the image.

Most real-world cases are blind, especially with spatially varying blur.

### 🧠 3. Deblurring Techniques

🔹 For Uniform Blur (Single PSF across entire image):

- Use Wiener filtering, Richardson–Lucy deconvolution, or regularized inverse filtering.
- These are frequency-domain or iterative techniques that attempt to undo convolution.

🔹 For Non-Uniform / Spatially Varying Blur:

- The image is often segmented into patches with locally uniform blur.
- Each region is deblurred using a locally estimated PSF.
- Methods include:

  - PSF estimation from edge profiles or image gradients.
  - Depth-aware blur modeling using depth maps (if available).
  - Layer-based approaches: decompose the scene into depth layers and deblur each with a different kernel.

#### 🧠 Example Algorithm

- Estimate a blur map (spatial distribution of PSF size or blur strength).
- Estimate PSFs in local regions.
- Apply spatially adaptive deconvolution:
- Either vary the filter across the image or use multi-scale, patch-based restoration.
- Blend the patches smoothly to avoid seams.

### 📸 Using Depth Information

If you have a depth map (from stereo, LiDAR, or depth sensor):

- Map depth values to blur kernel sizes.
- Use a depth-to-blur model (e.g., using camera aperture and lens parameters).
- Apply spatially varying deconvolution based on depth-derived PSFs.

### ⚠️ Challenges

- Estimating accurate blur kernels is hard, especially for real motion.
- Inverting convolution is unstable and amplifies noise → requires regularization.
- Edge ringing and artifacts are common.
