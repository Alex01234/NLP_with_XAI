# NLP_with_XAI
## Overview

In this project, I will describe step-by-step how to fine-tune a BERT model to classify texts with multiple labels and how to use the eXplainable AI (XAI) models LIME and SHAP in order to understand individual classifications of texts made by BERT. 

---
## Setup, installing Python and installing packages
For reference, I am using a PC, running Windows 10 Home 64-bit operating system. My machine has an 11th Gen Intel(R) Core(TM) i7-1195G7 @ 2.90GHz processor, 16 GB of RAM and a NVIDIA® GeForce RTX™ 3050 Laptop GPU, 4GB GDDR6 graphics card.

#### Downloading the data
For this project, [this](https://www.kaggle.com/datasets/vetrirah/janatahack-independence-day-2020-ml-hackathon) data from Kaggle has been used, which consists of research articles which has been classified with multiple labels. The labels represent the subjects that the research articles are associated with. Download the data (register a free account on Kaggle if you do not have one). From the downloaded zip-archive, select the file "train.csv" and rename it to "data.csv", and store it in a folder of your choosing. This is the dataset that we will create our training, validation and test-set from. 

#### Installing Python 
For this project, use Python 3.7.0. I installed it from here: https://www.python.org/downloads/release/python-370/, and chose "Windows x86-64 executable installer". Make sure to check "Add Python 3.7 to PATH". Verify that Python 3.7.0 is installed, on windows the command "python --version" should print "Python 3.7.0". 

#### Installing Python packages
- Install the package "virualenv" using pip. In cmd, run the command "pip install virtualenv". 
- 

---
## Exploring and preparing data
---
## Cross validation
---
## Fine-tuning/ Training
---
## Testing
---
## Generating explanations with LIME
---
## Generating explanations with SHAP
---

Note: This code is based upon an original code by myself (Alexander Dolk) and Hjalmar Davidsen. The [original code](https://github.com/Alex01234/MastersThesis/) was used for our Master thesis at Stockholm University, and for our [published article](https://ecp.ep.liu.se/index.php/shi/article/view/456). 
