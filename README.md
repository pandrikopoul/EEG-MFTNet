# EEG-MSTDNet: An EEGNet-Based Deep Learning Model with Multi-Scale Temporal Convolutions and Transformer Fusion for Motor Imagery Decoding
This project presents EEG-MSTDNet, a novel deep learning model
based on the EEGNet architecture, enhanced with multi-scale temporal convolutions and a
Transformer encoder stream. These components are designed to capture both short and long-
range temporal dependencies in EEG signals. The model is evaluated on the SHU dataset
using a subject-dependent cross-session setup.

![ModelPltFinal](https://github.com/user-attachments/assets/e5f2ce38-346f-447c-a4c0-eeffa65f455f)


## Install requirments:
pip install -r requirements.txt

## Data availability:
The dataset used in this project is publicly available at the following link: https://figshare.com/articles/code/shu_dataset/19228725. Please download the classification-ready .mat files, which have been preprocessed by the dataset creators.

## Quick Start:
1. Download the dataset from the link above.

2. Configure the dataset path inside the EEGMSTDNet_main.py file.

3. Run the model by executing: python EEGMSTDNet_main.py

To perform model interpretation, open the Jupyter notebook Model_interpretation_sub_6_Session_4.ipynb and run all the cells sequentially from the beginning. 
