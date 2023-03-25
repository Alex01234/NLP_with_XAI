# Natural Language Processing (Multi-label classification) with eXplainable Artifical Intelligence

In this project, I will describe step-by-step how to fine-tune a BERT model to classify texts with multiple labels and how to use the eXplainable AI (XAI) models LIME and SHAP in order to understand individual classifications of texts made by BERT. 

Note: This code is based upon an original code by myself (Alexander Dolk) and Hjalmar Davidsen. The [original code](https://github.com/Alex01234/MastersThesis/) was used for our Master thesis at Stockholm University, and for our [published academic article](https://ecp.ep.liu.se/index.php/shi/article/view/456). 

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
- Next, we need to create our datasets. These are a dataset that contains both the training set and validation set (which we will call "no_test.csv" and that we will use during cross validation), the training set, validation set and test set. Run the function create_training_validation_and_test_sets in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py create_training_validation_and_test_sets subset.csv no_test.csv train.csv validation.csv test.csv". This creates the files no_test.csv (which contains the same data as in the files train.csv and validation.csv, and that we will use during cross validation), train.csv (72% of the data, which we will use to fine-tune the model), validation.csv (18% of the data, which we will use for validation during the fine-tuning) and test.csv (10% of the data, which we will use to test the model).

![exploring_and_preparing_data](https://github.com/Alex01234/NLP_with_XAI/blob/main/exploring_and_preparing_data.PNG)

---
## Cross validation
All hyperparameters used to fine-tune the BERT model, except for the number of epochs, are based on ![academic studies](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ). Also, due to the technical limitaions of my machine, I was required to use a batch size of 1. In order to determine how many epochs to fine-tune (train) the model for, cross-validiation needs to be performed. The cross-validation is performed with the dataset that contains all data except for the test-data, meaning the training data and the validation data. This data is stored in the file "no_test.csv". Call the function cross_validation in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py cross_validation no_test.csv". This process may take many hours, depending on your machine. Model checkpoints are stored on your machine in the directory "output_directory". The output in the console describes the training loss and validation loss during the cross validation. I have copied and pasted this output to the document "cross_validation_output.txt", as well as arranged the results in the document "cross_validation_output.ods" (seen in the image below). Unfortunately, due to the lage output in the command prompt, the results from the first iteration were lost. As seen by the results, the lowest validation loss is achieved after only one epoch of fine-tuning, which will be used during the real fine-tuning of the model.

![cross_validation_results](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/cross_validation_results.PNG)

---
## Fine-tuning/ Training

To fine-tune the model, call the function fine_tune_model in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py fine_tune_model train.csv validation.csv output_dir_model_and_tokenizer fine_tuned_model fine_tuned_tokenizer". The fine-tuned model and tokenizer will be saved in the directories "\output_dir_model_and_tokenizer\fine_tuned_model" and "\output_dir_model_and_tokenizer\fine_tuned_tokenizer". 

![fine_tuning](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/fine_tuning.PNG)
---
## Testing

- To test the model against with test-set (the file test.csv), call the function test_model in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py test_model Full_path_to_fine-tuned_tokenizer Full_path_to_fined-tuned_model test.csv", example "python NLP_multi_label_classification.py test_model C:\Users\Alexander\source\nlp_with_xai\output_dir_model_and_tokenizer\fine_tuned_tokenizer C:\Users\Alexander\source\nlp_with_xai\output_dir_model_and_tokenizer\fine_tuned_model test.csv". This will test the fine-tuned model against the test-set, and ouput the accuracy, as well as the weighted, micro-averaged and macro-averaged precision, recall and F1 score. 
  
![testing_results](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/testing_results.PNG)
  
 - To test the BERT model with individual samples, call the function get_prediction_for_text in the file NLP_multi_label_classification.py. From cmd: "python NLP_multi_label_classification.py get_prediction_for_text Full_path_to_fine-tuned_tokenizer Full_path_to_fined-tuned_model Input_text_without_new_line_characters_enclosed_in_double_quotes". In the examples below from the test-set, you see the predicted probabilities by the BERT model, for each of the labels in the dataset. That is, for the labels Computer Science, Physics, Mathematics and Statistics.
 
- Example 1:

![BERT_test_1](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/BERT_test_1.PNG)

The predicted probabilites for example 1 are 0.99686, 0.000031286, 0.0027836 and 0.00032599 <br>
The true values for the labels are 1, 0, 1 and 0.

- Example 2: 

![BERT_test_2](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/BERT_test_2.PNG)

The predicted probabilites for example 2 are 0.88798, 0.000096437, 0.11174 and 0.00018281 <br>
The true values for the labels are 1, 0, 1 and 0.

- Example 3: 

![BERT_test_3](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/BERT_test_3.PNG)

The predicted probabilites for example 3 are 0.000056733, 0.00001346, 0.81582 and 0.18411 <br>
The true values for the labels are 0, 0, 1 and 1.

- Example 4: 

![BERT_test_4](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/BERT_test_4.PNG)

The predicted probabilites for example 4 are 0.000017749, 0.0000030367, 0.99998 and 0.0000042155 <br>
The true values for the labels are 1, 0, 1 and 0.

---
## Generating explanations with LIME
To generate explanations with LIME, we will use Google Colab Notebooks.
- Go to Google Drive.
- Either create a Google Colab Notebook and name it to whatever you want, or upload the file NLP_with_XAI_LIME.ipynb.
- To Google Drive, you will upload your folders containing the fine tuned model and fine tuned tokenizer that were generated when we trained our BERT model. Observe that the paths to the tokenizer and the model might have to be adjusted in the file NLP_with_XAI_LIME.ipynb
- Re-use the code from the file NLP_with_XAI_LIME.ipynb, either by downloading the file and uploading it to your Google Drive, or copying and pasting the code to your own Google Colab Notebook.
- Run the code, cell by cell.
- In the final cell, the variable "text" can be set equal to the text that you want to generate an explanation for. This will generate explanations for the classifications of all the labels by the BERT model, as interpreted by LIME. See section 3.6 in this [article](https://ecp.ep.liu.se/index.php/shi/article/view/456) for an explanation how LIME generates explanations.

- Example 1: 

![LIME_test_1](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/LIME_test_1.PNG)

For example 1, LIME predicts the label ComputerScience with a probability of 0.92, with the words "algorithm" and "complexity" having the greatest positive impact to the classification.

- Example 2: 

![LIME_test_2](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/LIME_test_2.PNG)

For example 2, LIME predicts the label ComputerScience with a probability of 0.99, with the words "code", "codeword", "alphabet" and "codes" having the greatest positive impact on the classification. Interestingly, the same words have the greates negative impact for the text to be labelled with the label Mathematics.

- Example 3: 

![LIME_test_3](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/LIME_test_3.PNG)

For example 3, LIME predicts the label Mathematics with a probability of 0.96, with the word "functions" having the greatest positive impact on the classification, although it seems no word has an impact much greater than many other words. 

- Example 4: 

![LIME_test_4](https://raw.githubusercontent.com/Alex01234/NLP_with_XAI/main/LIME_test_4.PNG)

For example 4, LIME predicts the label Mathematics with a probability of 1.00, with the words "axiomatisability" and "finite" having the greatest positive impact on the classification.

---
## Generating explanations with SHAP
To generate explanations with SHAP, we will use Google Colab Notebooks aswell.
- Go to Google Drive.
- Either create a Google Colab Notebook and name it to what you'd like, or upload the file NLP_with_XAI_SHAP.ipynb.
- Observe that you need the folders with the fine-tuned tokenizer and the fine-tuned model uploaded to Google Drive, as described in the section "Generating explanations with LIME". Similarly, the paths to the tokenizer and the model might have to be adjusted in the file NLP_with_XAI_SHAP.ipynb
- Re-use the code from the file NLP_with_XAI_SHAP.ipynb, either by downloading the file and uploading it to your Google Drive, or copying and pasting the code to your own Google Colab Notebook.
- Run the code, cell by cell.
- In the final cell, the variable "text" can be set equal to the text that you want to generate an explanation for. This will generate explanations for the classifications of all the labels by the BERT model, as interpreted by SHAP. See section 3.6 in this [article](https://ecp.ep.liu.se/index.php/shi/article/view/456) for an explanation how SHAP generates explanations.

- Example 1: 

gif1

Comment 1

- Example 2: 

gif2

Comment 2

- Example 3: 

gif3

Comment 3

- Example 4: 

gif4

Comment 4

---
