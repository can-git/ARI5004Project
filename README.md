
# Deep Learning Networks Comparison of Histologic Patterns on Resected Lung Adenocarcinoma Slides

This repository contains the code for the project article prepared for the ARI5004 (Deep Learning) course at Bahçeşehir University. 

NeurIPS styled report: [Deep Learning Networks Comparison of Histologic Patterns on Resected Lung
Adenocarcinoma Slides](https://drive.google.com/file/d/1UTf17-I8bDv6A56lEj0ev_KBVB1QGvyU/view?usp=sharing).

<p align="center">
<img src="etc/lung-cancer.jpg" height=350>
</p>

## Requirements

Python packages required (can be installed via pip or conda):

``` 
torchvision
tqdm
numpy
pandas
matplotlib
scikit-learn
seaborn
pillow
```

## Data Preparation

A proposed dataset can be achieved from [Drive](https://drive.google.com/file/d/1Q2SaakGyjvc5FaaaBisKaBBGwh0J5z0X/view?usp=sharing).

After downloading, dataset must be in the specific format as below:

```
Data
    │
    └── val
          ├── <classname1>
                          │ 
                          ├── <img1>
                          ├── <img2>
                          └── ...
          ├── <classname2>  
          └── ...    
      
```

## Additional Details

### In the Preprocess.py file:
``` python
df_classes = df_classes.iloc[:4000]
```
This line is added to fix the data imbalance between classes. 
Change for personal wishes or put a comment character(#) on the beginning to disable it.


## Hyperparameters
Models' important parameters can be adjusted from the terminal or Main.py file as default.

```
Options:
  --name TEXT            Name of the model. (cnn8, resnet18 or densenet121)
  --batch_size INTEGER   Batch Size
  --num_workers INTEGER  Num Workers
  --epochs INTEGER       Epochs
  --lr FLOAT             Learning Rate
  --wd INTEGER           Weight Decay
  --gamma FLOAT          Gamma
  --save BOOLEAN   Save Model at the end
  --im_size INTEGER      Image Size
```

## Training

To train the model, run this command or with the desired parameters as "--name":

```train
python Main.py --name <modelname>
```
This will create a Results folder, and model will be saved by the name value in the Folder.
<br />*Currently a few models are supported(cnn8, resnet18 and densenet121)*

## Evaluation

To evaluate model with lung dataset, ".pt" file should be as in the example:

```results
Results
       │
       └── <modelname>
                      └── <modelname>_model.pt
```
If the format is as above, then the code below will work successfully.

```eval
python Evaluation.py --name <modelname>
```
This will save some metric result images in the Results folder.

## Results

Our models achieve the following performances on :


| Model name                                                                                     | Accuracy | Precision | Recall | F1    |
|------------------------------------------------------------------------------------------------|----------|-----------|--------|-------|
| [CNN](https://drive.google.com/file/d/1Mhbxo7XYpfiaX-KtvCK_PL4HfoGSjWqa/view?usp=sharing)      | 0.89%    | 0.89%     | 0.88%  | 0.89% |
| [ResNet](https://drive.google.com/file/d/1sWPm_AEl2mOrgICo2xV8whX_ht7wYYn3/view?usp=sharing)   | 0.92%    | 0.93%     | 0.92%  | 0.92% |
| [DenseNet](https://drive.google.com/file/d/1xIVPKVo8dyuEyHEAa04DCTCnOj-mcZ_R/view?usp=sharing) | 0.94%    | 0.94%     | 0.94%  | 0.94% |


## Contributing
This project is prepared for the ARI5004 (Deep Learning) course at Bahçeşehir University. 
Thank you to my professor Mustafa Umit Oner for all the achievements.

Dataset and idea is borrowed from [Deepslide](https://github.com/BMIRDS/deepslide), Thanks for their excellent work!
