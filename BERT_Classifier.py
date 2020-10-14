#!/usr/bin/env python
# coding: utf-8

# # BERT implementation in PyTorch

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch.optim as optim
import numpy as np
from tensorflow.keras.models import model_from_json
# from onnx2keras import onnx2keras
import cv2
import os,sys,re
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.model_selection import StratifiedKFold
from text_preprocessing import *
from testing import *
from transformers import BertConfig
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import statistics
import time
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# In[2]:

torch.manual_seed(42)

def make_bert_input(data, max_len):
    # For every sentence...
    input_ids = []
    attention_masks = []
    token_ids = []
    for sent in data:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_token_type_ids = True,
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        token_ids.append(encoded_dict['token_type_ids'])

    input_ids = np.asarray(input_ids, dtype='int32')
    attention_masks = np.asarray(attention_masks, dtype='int32')
    token_ids = np.asarray(token_ids, dtype='int32')

    new_data = np.concatenate((input_ids, attention_masks), axis = 1)
    new_data = np.concatenate((new_data, token_ids), axis=1)

    return new_data

# Standard dataset class for pytorch dataloaders

# In[3]:


class Dataset(torch.utils.data.Dataset):
  # 'Characterizes a dataset for PyTorch'
    def __init__(self, text_input, text_mask, labels):
        'Initialization'
        self.labels = labels
        self.text_input = text_input
        self.text_mask = text_mask

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = np.vstack((self.text_input[index], self.text_mask[index]))
        y = self.labels[index]

        return X, y

# Defining BERT model using both OCR and Text inputs

# In[4]:


class Bert_Text_OCR(nn.Module):
  
    def __init__(self, num_labels, config=None, device=torch.device("cuda:0")):
        super(Bert_Text_OCR, self).__init__()
        self.bert_text = BertModel.from_pretrained('bert-base-uncased', config=config)
        self.bn = nn.BatchNorm1d(768, momentum=0.99)
        self.dense1 = nn.Linear(in_features=768, out_features=192) #Add ReLu in forward loop
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features=192, out_features=num_labels) #Add softmax in forward loop
        self.device = device
        
    def forward(self, inputs, attention_mask=None, labels=None):

        text_input_ids_in = inputs[:,0,:].long().to(self.device)
        text_input_masks_in = inputs[:,1,:].long().to(self.device)
        
        text_embedding_layer = self.bert_text(text_input_ids_in, attention_mask=text_input_masks_in)[0]
        text_cls_token = text_embedding_layer[:,0,:]
        X = self.bn(text_cls_token)
        X = F.relu(self.dense1(X))
        X = self.dropout(X)
        X = F.log_softmax(self.dense2(X))
        return X
        

# In[5]:


def save_models(epochs, model):
    torch.save(model.state_dict(), "bert_model_fold_{}.h5".format(epochs))
    print("Checkpoint Saved")

# In[6]:


def train_loop(dataloaders, dataset_sizes,  num_classes, config=None, epochs=1):
    model = Bert_Text_OCR(num_labels=num_classes, config=config)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-08) # clipnorm=1.0, add later
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
#                 scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    actual_labels = torch.max(labels.long(), 1)[1]
                    loss = criterion(outputs, actual_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
#                 running_loss += loss.item() * inputs.size(0)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == actual_labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_loss < best_loss:
#                 save_models(epoch,model)
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[7]:


datafile = "Abuse_Analyzer_new.tsv"
data_col = 0
label_col = 2
max_len = 100
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_cum = 0
rec_cum = 0
pre_cum = 0
f1_cum = 0
f1_cum_mic = 0
acc_arr = []
rec_arr = []
pre_arr = []
f1_arr = []
f1_arr_mic = []
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)

#------------------------------------------------------------------------------------------------
text_data, labels = prepare_dataset(datafile, data_col, label_col, "word-based")

print("Number of Examples: ", len(text_data))

encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
class_weights_labels = class_weight.compute_class_weight('balanced',
                                             np.unique(encoded_labels),
                                             encoded_labels)

num_classes = len(list(encoder.classes_))
print("num_classes: ", num_classes)
print(encoder.classes_)
config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states = False


fold_number = 1

new_data = make_bert_input(text_data, max_len)

## Add image input to new_data, flatten images then unflatten later

encoded_labels = np.asarray(encoded_labels, dtype='int32')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weights_labels = torch.tensor(class_weights_labels, dtype=torch.float, device=device)

for train_index, test_index in skf.split(new_data, encoded_labels):
    print("Running fold #", fold_number)
    train_data, test_data = new_data[train_index], new_data[test_index]
    train_label, test_label = encoded_labels[train_index], encoded_labels[test_index]
    train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2, random_state=42)
    
    train_label = to_categorical(train_label)
    validation_label = to_categorical(validation_label)
    metric_test = np.copy(test_label)
    test_label = to_categorical(test_label)

    train_text_input_ids = np.copy(train_data[:,0:max_len])
    validation_text_input_ids = np.copy(validation_data[:,0:max_len])
    test_text_input_ids = np.copy(test_data[:,0:max_len])
    train_text_attention_mask = np.copy(train_data[:,max_len:2*max_len])
    validation_text_attention_mask = np.copy(validation_data[:,max_len:2*max_len])
    test_text_attention_mask = np.copy(test_data[:,max_len:2*max_len])

    training_set = Dataset(train_text_input_ids, train_text_attention_mask, train_label)
    validation_set = Dataset(validation_text_input_ids, validation_text_attention_mask, validation_label)
    test_set = Dataset(test_text_input_ids, test_text_attention_mask, test_label)

    dataloaders = {
        'train' : torch.utils.data.DataLoader(training_set, batch_size=4,
                                             shuffle=True, num_workers=2, drop_last=True),
        'validation' : torch.utils.data.DataLoader(validation_set, batch_size=4,
                                             shuffle=True, num_workers=2, drop_last=True)
    }

    dataset_sizes = {
        'train': len(training_set),
        'validation': len(validation_set),
    }

    model = train_loop(dataloaders, dataset_sizes, num_classes, config=config, epochs=15)
    #         save_models(fold_number, model)

    #new_test_data = Dataset(test_data[:,0:max_len], test_data[:,max_len:2*max_len], test_data[:,3*max_len:4*max_len], test_data[:,4*max_len:5*max_len], test_label)

    y_pred = np.array([])

    for i in tqdm(range(len(test_set))):
        inputs = torch.Tensor([test_set[i][0]]).to(device)
        model.eval()
        outputs = model(inputs)
        preds = torch.max(outputs, 1)[1]
        y_pred = np.append(y_pred, preds.cpu().numpy())

    acc_arr.append(accuracy_score(metric_test, y_pred))
    acc_cum += acc_arr[fold_number-1]
    rec_arr.append(recall_score(metric_test, y_pred, average='macro'))
    rec_cum += rec_arr[fold_number-1]
    pre_arr.append(precision_score(metric_test, y_pred, average='macro'))
    pre_cum += pre_arr[fold_number-1]
    f1_arr.append(f1_score(metric_test, y_pred, average='macro'))
    f1_cum  += f1_arr[fold_number-1]
    f1_arr_mic.append(f1_score(metric_test, y_pred, average='micro'))
    f1_cum_mic  += f1_arr_mic[fold_number-1]
    fold_number+=1

print("Accuracy: ", acc_cum/5)
print("Recall: ", rec_cum/5)
print("Precision: ", pre_cum/5)
print("F1 score: ", f1_cum/5)
print("F1 score Micro: ", f1_cum_mic/5)

print("------------------------------")
print("Accuracy_stdev: ", statistics.stdev(acc_arr))
print("Recall_stdev: ", statistics.stdev(rec_arr))
print("Precision_stdev: ", statistics.stdev(pre_arr))
print("F1 score_stdev: ", statistics.stdev(f1_arr))
print("F1 score_stdev Micro: ", statistics.stdev(f1_arr_mic))


# In[ ]: