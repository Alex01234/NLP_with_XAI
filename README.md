# Natural Language Processing (Multi-label classification) with eXplainable Artifical Intelligence

In this project, I will describe step-by-step how to fine-tune a BERT model to classify texts with multiple labels and how to use the eXplainable AI (XAI) models LIME and SHAP in order to understand individual classifications of texts made by BERT. 

---
## Setup: Downloading data, installing Python, setting up virtual environment and installing packages
For reference, I am using a PC, running Windows 10 Home 64-bit operating system. My machine has an 11th Gen Intel(R) Core(TM) i7-1195G7 @ 2.90GHz processor, 16 GB of RAM and a NVIDIA® GeForce RTX™ 3050 Laptop GPU, 4GB GDDR6 graphics card.

#### Downloading the data
For this project, [this](https://www.kaggle.com/datasets/vetrirah/janatahack-independence-day-2020-ml-hackathon) data from Kaggle has been used, which consists of research articles which has been classified with multiple labels. The labels represent the subjects that the research articles are associated with. Download the data (register a free account on Kaggle if you do not have one). From the downloaded zip-archive, select the file "train.csv" and rename it to "data.csv", and store it in a folder of your choosing. This is the dataset that we will create our training, validation and test-set from. 

#### Installing Python 
For this project, use Python 3.7.0. I installed it from here: https://www.python.org/downloads/release/python-370/, and chose "Windows x86-64 executable installer". Make sure to check "Add Python 3.7 to PATH". Verify that Python 3.7.0 is installed, on windows the command "python --version" should print "Python 3.7.0". 

#### Setting up virtual environment
- Install the package "virualenv" using pip. In cmd, run the command "pip install virtualenv". 
- In cmd, navigate to a directory of your choice. Create a directory (which we will be working in) using the mkdir command. Type "mkdir <directory_name>". Navigate into the newly created directory. Create a virtual enviornment with the command "virtualenv env". To activate the virtual environment, run the command "env\Scripts\activate.bat". This command must be run from this directory every time you want to activate the virtual environment, which we will be working in.
- Also, move the file "data.csv" to the directory that you have created.
![setting_up_virtual_environment](https://github.com/Alex01234/NLP_with_XAI/blob/main/setting_up_virtual_environment.PNG)

#### Installing Python packages
- Install the packages torch, torchvision and torchaudio with cuda enabled. Run the command: "pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html" 
![installing_python_packages_1](https://github.com/Alex01234/NLP_with_XAI/blob/main/installing_python_packages_1.PNG)

- Install the remaining of the required packages from the file ["required_packages.txt"](https://github.com/Alex01234/NLP_with_XAI/blob/main/required_packages.txt). Place the file required_packages.txt in the directory and run the command "pip install -r required_packages.txt". 
![installing_python_packages_2](https://github.com/Alex01234/NLP_with_XAI/blob/main/installing_python_packages_2.PNG)

---
## Exploring and preparing data
- First, the data (data.csv) needs to be cleaned. The texts in the column "ABSTRACT" contains new-line characters, which we need to replace with spaces. This will make it easier for test the model later. I am using Libre Office Calc to do this. Open the file data.csv with Libre Office Calc, mark the column "ABSTRACT", select the "Edit" tab, select "Find and Replace...", enter "\n" in the "Find" field, enter " " (space character) in the "Replace" field, tick "Current selection only" and "Regular expressions" boxes. Click "Replace all". The changes should take place. Save the file (Use text CSV format).
- Visualize the class distribution in the data, by running the function visualise_class_distribution in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py visualise_class_distribution data.csv"
![data_class_distribution.png](https://github.com/Alex01234/NLP_with_XAI/blob/main/data_class_distribution.png)
- As visible in the plot above, the classes "Quantitative Biology" and "Quantitative Finance" (the two classes most to the right) are very underrepresented in the data. When we fine-tune our model we want to have data with a lot of samples in all classes. In a production scenario, we might use sampling techniques like oversampling or undersampling to generate a dataset with a better balance between the classes. For this project, we will simply drop all samples from the dataset that are not labeled as "true" for one of the adequately represented classes (Computer Science, Physics, Mathematics and Statistics). 
- Call the function create_subset_of_data in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py create_subset_of_data data.csv subset.csv". This creates the file "subset.csv" in the working directory, which only has samples that have the value true in one or more of the classes Computer Science, Physics, Mathematics and Statistics. Furthermore, only those labels have been kept in the dataset.
- 
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
