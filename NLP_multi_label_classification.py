import sys
import os
import torch
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, BertModel, BertForSequenceClassification, TrainingArguments, Trainer)
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
from statistics import mean
from sklearn.model_selection import KFold
import operator
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

def get_dictionary_from_df(df, names):
    dictionary = {}
    for col in names:
        dictionary[col] = (df[col] == 1).sum()

    sorted_dict = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_dict

def display_dictionary(dictionary):
    sorted_list_of_tuples = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    x, y = zip(*sorted_list_of_tuples)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.xticks(rotation=90)
    plt.show()

def visualise_class_distribution(dataset_name):
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    data = pd.read_csv(os.getcwd() + "\\" + dataset_name,delimiter=',')

    # Look at distribution of full dataframe
    full_names = 'Computer Science,Physics,Mathematics,Statistics,Quantitative Biology,Quantitative Finance'
    full_names = full_names.split(",")
    data_only_labels = data.drop(['ID'], axis=1)
    data_only_labels = data_only_labels.drop(['TITLE'], axis=1)
    data_only_labels = data_only_labels.drop(['ABSTRACT'], axis=1)


    sorted_dict = get_dictionary_from_df(data_only_labels, full_names)
    print("sorted_dict:")
    print(sorted_dict)
    display_dictionary(sorted_dict)

def create_subset_of_data(file_name_input_data, file_name_resulting_data):

    data = pd.read_csv(os.getcwd() + "\\" + file_name_input_data, delimiter=',')

    d = data.loc[((data['Computer Science'] == 1) | (data['Physics'] == 1) | (data['Mathematics'] == 1) | (data['Statistics'] == 1))]
    print(d.shape)

    # show distribution of d
    d_only_labels = d.drop(['ID'], axis=1)
    d_only_labels = d_only_labels.drop(['TITLE'], axis=1)
    d_only_labels = d_only_labels.drop(['ABSTRACT'], axis=1)
    print(d_only_labels)
    full_names = 'Computer Science,Physics,Mathematics,Statistics,Quantitative Biology,Quantitative Finance'
    full_names = full_names.split(",")
    d_dict = get_dictionary_from_df(d_only_labels, full_names)
    print("d_dict:")
    print(d_dict)

    final_table_columns = ['ID', 'TITLE', 'ABSTRACT', 'Computer Science', 'Physics', 'Mathematics', 'Statistics']
    d1 = d[d.columns.intersection(final_table_columns)]

    d1.to_csv(os.getcwd() + "\\" + file_name_resulting_data,index=False)

def cross_validation(file_name_data_without_test):

    print(os.getcwd() + "\\" + file_name_data_without_test)

    
    filepath_train = os.getcwd() + "\\" + file_name_data_without_test
    data_train = pd.read_csv(filepath_train, delimiter=',')
    label_cols = [c for c in data_train.columns if c not in ["ID", "TITLE", "ABSTRACT"]]
    print(label_cols)
    data_train["labels"] = data_train[label_cols].values.tolist()

    
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )
    
    # prepare cross validation
    n = 5
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    iteration = 0
    for train_index, val_index in kf.split(data_train):
        iteration = iteration + 1
        print("Iteration", iteration)
        # splitting Dataframe (dataset not included)
        train_df = data_train.iloc[train_index]
        val_df = data_train.iloc[val_index]

        train_encodings = tokenizer(train_df["ABSTRACT"].values.tolist(), truncation=True)
        val_encodings = tokenizer(val_df["ABSTRACT"].values.tolist(), truncation=True)

        train_labels = train_df["labels"].values.tolist()
        val_labels = val_df["labels"].values.tolist()

        val_labels_as_floats = []
        for list in val_labels:
            new_inner_list = []
            for number in list:
                new_float = float(number)
                new_inner_list.append(new_float)
            val_labels_as_floats.append(new_inner_list)

        train_dataset = Dataset(train_encodings, train_labels)
        val_dataset = Dataset(val_encodings, val_labels_as_floats)

        num_labels = 4
        model = BertForMultilabelSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification", num_labels=num_labels).to('cuda')

        batch_size = 1
        logging_steps = len(train_dataset) // batch_size

        args = TrainingArguments(
            output_dir="output_directory",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            gradient_accumulation_steps=1,
            warmup_steps=155,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_steps=logging_steps
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer)

        metrics = trainer.evaluate()
        print(metrics)

        trainer.train()
        metrics2 = trainer.evaluate()
        print(metrics2)

def fine_tune_model(file_name_training_data, file_name_validation_data, folder_output_directory, name_output_model, name_output_tokenizer):
    filepath_train = os.getcwd() + "\\" + file_name_training_data
    data_train = pd.read_csv(filepath_train, delimiter=',')
    label_cols = [c for c in data_train.columns if c not in ["ID", "TITLE", "ABSTRACT"]]
    data_train["labels"] = data_train[label_cols].values.tolist()
    filepath_validation = os.getcwd() + "\\" + file_name_validation_data
    data_validation = pd.read_csv(filepath_validation, delimiter=',')
    label_cols_val = [c for c in data_validation.columns if c not in ["ID", "TITLE", "ABSTRACT"]]
    data_validation["labels"] = data_validation[label_cols_val].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_encodings = tokenizer(data_train["ABSTRACT"].values.tolist(), truncation=True)
    val_encodings = tokenizer(data_validation["ABSTRACT"].values.tolist(), truncation=True)

    train_labels = data_train["labels"].values.tolist()
    val_labels = data_validation["labels"].values.tolist()

    test_labels_as_floats = []
    for list in val_labels:
        new_inner_list = []
        for number in list:
            new_float = float(number)
            new_inner_list.append(new_float)
        test_labels_as_floats.append(new_inner_list)

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, test_labels_as_floats)

    num_labels = 4
    model = BertForMultilabelSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification", num_labels=num_labels).to('cuda')

    batch_size = 1
    logging_steps = len(train_dataset) // batch_size

    args = TrainingArguments(
        output_dir="output_directory",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        gradient_accumulation_steps=1,
        warmup_steps=155,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=logging_steps
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer)

    metrics = trainer.evaluate()
    print(metrics)

    trainer.train()
    metrics2 = trainer.evaluate()
    print(metrics2)

    model_dir = os.getcwd() + "\\" + folder_output_directory
    model.save_pretrained(model_dir + "\\" + name_output_model)
    tokenizer.save_pretrained(model_dir + "\\" + name_output_tokenizer)

def test_model(full_path_fine_tuned_tokenizer, full_path_fine_tuned_model, file_name_test_data):
    tokenizer = AutoTokenizer.from_pretrained(
        full_path_fine_tuned_tokenizer,
        local_files_only=True,
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )

    model1 = AutoModelForSequenceClassification.from_pretrained(
        full_path_fine_tuned_model, local_files_only=True,
        problem_type="multi_label_classification", num_labels=4)

    test_data = pd.read_csv(os.getcwd() + "\\" + file_name_test_data)

    label_cols_test = [c for c in test_data.columns if c not in ["ID", "TITLE", "ABSTRACT"]]

    test_data["labels"] = test_data[label_cols_test].values.tolist()

    true_values = []
    predictions = []

    for index, row in test_data.iterrows():
        abstract = row['ABSTRACT']
        correct_labels = row['labels']
        predictions_as_probs = get_prediction(abstract, model1, tokenizer)
        pred = calc_threshold(predictions_as_probs)
        true_values.append(correct_labels)
        predictions.append(pred)

    acc = []

    prec_we = []
    rec_we = []
    f1_we = []

    prec_mic = []
    rec_mic = []
    f1_mic = []

    prec_mac = []
    rec_mac = []
    f1_mac = []

    for true_val, pred in zip(true_values, predictions):
        acc.append(accuracy_score(y_true=true_val, y_pred=pred))

        prec_we.append(precision_score(y_true=true_val, y_pred=pred, average='weighted'))
        rec_we.append(recall_score(y_true=true_val, y_pred=pred, average='weighted'))
        f1_we.append(f1_score(y_true=true_val, y_pred=pred, average='weighted'))

        prec_mic.append(precision_score(y_true=true_val, y_pred=pred, average='micro'))
        rec_mic.append(recall_score(y_true=true_val, y_pred=pred, average='micro'))
        f1_mic.append(f1_score(y_true=true_val, y_pred=pred, average='micro'))

        prec_mac.append(precision_score(y_true=true_val, y_pred=pred, average='macro'))
        rec_mac.append(recall_score(y_true=true_val, y_pred=pred, average='macro'))
        f1_mac.append(f1_score(y_true=true_val, y_pred=pred, average='macro'))

    # Calculate mean acc, prec, rec, f1 from the lists and print it
    print('Mean accuracy:')
    print(mean(acc))

    print('Mean precision weighted:')
    print(mean(prec_we))
    print('Mean recall weighted:')
    print(mean(rec_we))
    print('Mean F1 weighted:')
    print(mean(f1_we))

    print('Mean precision micro-averaged:')
    print(mean(prec_mic))
    print('Mean recall micro-averaged:')
    print(mean(rec_mic))
    print('Mean F1 micro-averaged:')
    print(mean(f1_mic))

    print('Mean precision macro-averaged:')
    print(mean(prec_mac))
    print('Mean recall macro-averaged:')
    print(mean(rec_mac))
    print('Mean F1 macro-averaged:')
    print(mean(f1_mac))

def get_prediction(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs


def calc_threshold(predictions_as_probs):
    list_of_predictions_as_zeroes_and_ones = []
    a = []
    a = predictions_as_probs[0]
    listus = a.tolist()
    for value in listus:
        if value > 0.5:
            list_of_predictions_as_zeroes_and_ones.append(1)
        else:
            list_of_predictions_as_zeroes_and_ones.append(0)

    return list_of_predictions_as_zeroes_and_ones

def get_prediction_for_text(full_path_tokenizer, full_path_model, input_text):
    tokenizer = AutoTokenizer.from_pretrained(
        full_path_tokenizer,
        local_files_only=True,
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )
    model1 = AutoModelForSequenceClassification.from_pretrained(
        full_path_model,
        local_files_only=True,
        problem_type="multi_label_classification", num_labels=4)

    print(get_prediction(input_text, model1, tokenizer))


def create_training_validation_and_test_sets(file_name_input_data, file_name_data_without_test, file_name_data_train, file_name_data_validation, file_name_data_test):
    data = pd.read_csv(os.getcwd() + "\\" + file_name_input_data)
    data_without_test, test = train_test_split(data, test_size=0.1, random_state=42)
    train, validation = train_test_split(data_without_test, test_size=0.2, random_state=42)

    data_without_test.to_csv(os.getcwd() + "\\" + file_name_data_without_test, index=False)
    train.to_csv(os.getcwd() + "\\" + file_name_data_train, index=False)
    validation.to_csv(os.getcwd() + "\\" + file_name_data_validation, index=False)
    test.to_csv(os.getcwd() + "\\" + file_name_data_test, index=False)

def create_small_data_set(file_name_input_data, file_name_output_data):
    data = pd.read_csv(os.getcwd() + "\\" + file_name_input_data)
    non_selected_data, selected_data = train_test_split(data, test_size=0.2, random_state=42)
    selected_data.to_csv(os.getcwd() + "\\" + file_name_output_data, index=False)

if __name__ == "__main__":

    if sys.argv[1] == "create_small_data_set":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py create_small_data_set data.csv small_data.csv
        create_small_data_set(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "visualise_class_distribution":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py visualise_class_distribution data.csv
        visualise_class_distribution(sys.argv[2])
    if sys.argv[1] == "create_subset_of_data":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py create_subset_of_data data.csv subset.csv
        create_subset_of_data(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "create_training_validation_and_test_sets":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py create_training_validation_and_test_sets subset.csv no_test.csv train.csv validation.csv test.csv
        create_training_validation_and_test_sets(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    if sys.argv[1] == "cross_validation":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py cross_validation no_test.csv
        cross_validation(sys.argv[2])
    if sys.argv[1] == "fine_tune_model":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py fine_tune_model train.csv validation.csv output_dir_model_and_tokenizer fine_tuned_model fine_tuned_tokenizer
        fine_tune_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    if sys.argv[1] == "test_model":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py test_model 
        #C:\Users\Alexander\source\nlp_with_xai\output_dir_model_and_tokenizer\fine_tuned_tokenizer 
        #C:\Users\Alexander\source\nlp_with_xai\output_dir_model_and_tokenizer\fine_tuned_model 
        #test.csv
        test_model(sys.argv[2], sys.argv[3], sys.argv[4])
    if sys.argv[1] == "get_prediction_for_text":
        # (env) C:\Users\Alexander\source\nlp_with_xai>python NLP_multi_label_classification.py get_prediction_for_text 
        #C:\Users\Alexander\source\nlp_with_xai\output_dir_model_and_tokenizer\fine_tuned_tokenizer 
        #C:\Users\Alexander\source\nlp_with_xai\output_dir_model_and_tokenizer\fine_tuned_model 
        #<INPUT TEXT> (without new line characters, encased in double quotes)
        get_prediction_for_text(sys.argv[2], sys.argv[3], sys.argv[4])
