
# Deep Learning Networks Comparison of Histologic Patterns on Resected Lung Adenocarcinoma Slides

This repository contains the code for the project article prepared for the ARI5004 (Deep Learning) course at Bahçeşehir University. 

NeurIPS styled report: [Deep Learning Networks Comparison of Histologic Patterns on Resected Lung
Adenocarcinoma Slides](https://drive.google.com/file/d/1fs-4PK2iSX_sbzeuIlnXxy6YzBW0D-fZ/view?usp=sharing).

<p align="center">
<img src="etc/lung-cancer.jpg" height=350>
</p>

## Requirements

Python packages required (can be installed via pip or conda):

``` 
 - torchvision
 - tqdm
 - numpy
 - pandas
 - matplotlib
 - scikit-learn
 - seaborn
 - pillow
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
This will change Data folder as:
```
Data
    │
    ├── Lung_Data
                 ├──  <img1>
                 ├──  <img1>
                 └──  ...
    └── data.csv
```
This format is a must for train and evaluation processes.

## Hyperparameters

### In the Preprocess.py file:
``` python
df_classes = df_classes.iloc[:4000]
```
This line is added to fix the data imbalance between classes. 
Change for personal wishes or put a comment character(#) on the beginning to disable it.

### Adjusting Model Parameters in the Properties.py

```python
BATCH_SIZE = 1
NUM_WORKERS = 24
EPOCHS = 10
LR = 0.00008  # Learning Rate
WD = 0  # Weight Decay
GAMMA = 0.9
SAVE_MODEL = True
IMAGE_SIZE = 228
TRAIN = True  # Change this to False to analyse model(s) by some metrics
NUM_CLASSES = 5  # Amount of classes to classify in the model according to the dataset
```

## Training

To train the model, run this command:

```train
python Main.py --name <modelname>
```
*Currently a few models are supported(cnn8, resnet18 and densenet121)*

## Evaluation

To evaluate model with Lung_Data dataset, ".pt" file should be as in the example:

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

## Results

Our models achieve the following performances on :


| Model name                                                                                     | Accuracy | Precision | Recall | F1  |
|------------------------------------------------------------------------------------------------|----------|-----------|--------|-----|
| [CNN](https://drive.google.com/file/d/1nAc6Fbh0K4zq_njjYAkpSzHu4jhqq2xr/view?usp=sharing)      | 85%      | 95%       | 95%    | 95% |
| [ResNet](https://drive.google.com/file/d/1jktw8YApfWIJEpR-E45mdfThctAEr6si/view?usp=sharing)   | 85%      | 95%       | 95%    | 95% |
| [DenseNet](https://drive.google.com/file/d/1752e-nGk6Q6zinhugwBh6Ecf2rLB0aDG/view?usp=sharing) | 85%      | 95%       | 95%    | 95% |


## Known Issues and Limitations

```
 - Only 1 GPU is supported
```

## Contributing

Dataset and idea is borrowed from [Deepslide](https://github.com/BMIRDS/deepslide), Thanks for their excellent work!
