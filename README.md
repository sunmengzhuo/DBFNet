# DBFNet
## Requirements
* python3.11.4
* pytorch2.1.0+cu121
* tensorboard 2.14.0
## Usage
### 1.dataset
* Dual-Phase Tumor CT images, Lessor omental adipose tissue CT images, and clinical information in 483 patients with HCC.  
* **PS:** The data **cannot be shared publicly** due to the privacy of individuals that participated in the study and because the data is intended for future research purposes.
### 2.Train the DBFNet
* You need to train the DBFNet with the following commands:  
`$ python train.py`  
* You can modify the training hyperparameters in `$ train.py`.
### 4.Predict MVI
* If you wish to see predictions for DBFNet, you should run the following file:  
`$ python predict.py`
