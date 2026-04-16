# EEEG-MFTNet: An Enhanced EEGNet Architecture with Multi-Scale Temporal Convolutions and Transformer Fusion for Cross-Session Motor Imagery Decoding
This project presents EEG-MFTNet, a novel deep learning model
based on the EEGNet architecture, enhanced with multi-scale temporal convolutions and a
Transformer encoder stream. These components are designed to capture both short and long-
range temporal dependencies in EEG signals. The model is evaluated on the SHU dataset
using the subject-dependent cross-session setup provided by the dataset creators.

![cleanfinalmodelplot](https://github.com/user-attachments/assets/7682abc0-d7a8-4bd0-943b-0865952c11fb)


## ⚙️ Install Requirements:
To install the required dependencies, run:
```bash
pip install -r requirements.txt 
```
## 📂 Data Availability:
The dataset used in this project is publicly available at the following link: https://figshare.com/articles/code/shu_dataset/19228725. Please download the classification-ready .mat files, which have been preprocessed by the dataset creators.

## 🚀 Quick Start

The entire pipeline—from reading the data to training and evaluating the model—is contained within the `main` function of the `EEGMSTDNet_main.py` file. If you wish to skip the training process, pre-trained model weights for each subject are available in the Saved_Model_Weights folder.

1. Download the dataset from the link provided in the **📂 Data Availability** section.

2. Open `EEGMSTDNet_main.py` and set the correct path to the dataset directory.

3. Run the model by executing:
```bash
 python EEGMSTDNet_main.py

```


4. To conduct model interpretation and assess the trustworthiness of the interpretation method, open the Jupyter notebook Model_interpretation_sub_6_Session_4.ipynb and execute all cells sequentially from the beginning.  If you prefer to skip model training, you can use the pre-trained weights provided in the Interpretation_model.weights.h5 file.

## Citation

If you use this work, please cite:

```bibtex
@misc{andrikopoulos2026eegmftnetenhancedeegnetarchitecture,
  title={EEG-MFTNet: An Enhanced EEGNet Architecture with Multi-Scale Temporal Convolutions and Transformer Fusion for Cross-Session Motor Imagery Decoding}, 
  author={Panagiotis Andrikopoulos and Siamak Mehrkanoon},
  year={2026},
  eprint={2604.05843},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.05843}, 
}
```

