## U-ViL: A U-shaped Vision-LSTM Framework for Cardiac Image Segmentation

<p align="center"> <img src="./figs/u-vil.png" alt="U-ViL Architecture" width="600"/> </p>

[Full Paper (PDF)](https://nickdisanto.github.io/assets/pdfs/U-ViL.pdf)

Deep learning techniques have demonstrated remarkable success in medical image segmentation, but challenges remain in simultaneously capturing both global contextual dependencies and local structural details in the presence of structural variability. To address these challenges, we propose **U-ViL** (U-Net-like Vision-LSTM), a novel U-shaped architecture fused with Vision Long Short-Term Memory units. U-ViL incorporates Vision-LSTM blocks as the backbone of an encoder-decoder framework, aiming to model both low-level features and long-range dependencies. We evaluate the proposed model on the Automated Cardiac Diagnosis Challenge dataset and benchmark it against widely used segmentation architectures, including the conventional U-Net and the transformer-based Swin-Unet. Although the current implementation of U-ViL does not yield overall superior segmentation accuracy, it reveals distinct qualitative feature representations and segmentation patterns compared to prevailing architectures and could benefit from further hierarchical refinement. These findings highlight the potential for implementing a unified framework combining recurrent mechanisms that capture long-range spatial dependencies with localized spatial precision for medical image analysis.

You can download the **ACDC dataset** from [this Google Drive link](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view).

<p align="center"> <img src="./figs/acdc_samples.png" alt="ACDC Sample Images" width="400"/> </p>

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/NickDiSanto/U-ViL.git
   cd U-ViL/code
   ```

2. Training:
  ```python
  python train_fully_supervised_2D.py \
    --model uvil \
    --max_iterations 10000 \
    --batch_size 4 \
    --num_classes 4 \
    --exp ACDC/Fully_Supervised
  ```

3. Testing:
  ```python
  python test_2D_fully.py \
    --model uvil \
    --exp ACDC/Fully_Supervised
  ```

---

## Results

<h3 align="center">Quantitative</h3>
<p align="center">
  <img src="./figs/quantitative.png" alt="Quantitative Results" width="600"/>
</p>

<h3 align="center">Qualitative</h3>
<p align="center">
  <img src="./figs/visualizations.png" alt="Qualitative Visualizations" width="600"/>
</p>

---

## Contact
Ehsan Khodapanah Aghdam — ehsan.khodapanah.aghdam [at] vanderbilt.edu

Nick DiSanto — nicolas.c.disanto [at] vanderbilt.edu

---

## Acknowledgements
We gratefully acknowledge the authors of the following repositories, which inspired or contributed to this work:

- [Vision-LSTM](https://github.com/NX-AI/vision-lstm)

- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

- [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet/tree/main)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
