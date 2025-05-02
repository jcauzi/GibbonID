import os
import numpy as np
import tqdm
import datasets
from datasets import DatasetDict, Features
from datasets import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import csv
import pickle

##########################################################################################
####################################Arguments############################################# 
##########################################################################################


parser = argparse.ArgumentParser(description='Specify arguments Speech probe')

parser.add_argument('-d','--dataset', help='The dataset [10_50, 25_10, noise, pulse]',required=True)
parser.add_argument('-m','--model', help='The pre-trained model name [Unispeech, HuBERT]',required=True)
parser.add_argument('-e','--epochs', help='number of epochs',required=True, type=int)
parser.add_argument('-r', '--save_results', action='store_true', help='Wether to save results as new line in results.csv')
parser.add_argument('-l', '--layer', help='The embedding layer to be probed', required=True)

args = parser.parse_args()

##########################################################################################
##################################Dataset preparation##################################### 
##########################################################################################



source_path = "mount/Gibbon_ID_probing"
emb_dir = os.path.join(source_path, f"embeddings/{args.model}/layer_{args.layer}/embedding_pickles_{args.dataset}/train")

collection_train = {}
for pickled_emb in os.listdir(emb_dir) :
    if not pickled_emb.startswith('.') :
        with open(os.path.join(emb_dir, pickled_emb), 'rb') as f :
            emb = pickle.load(f)
        collection_train.update(emb)

if not args.dataset == "10_50" :
    source_path = "mount/Gibbon_ID_probing"
    emb_dir = os.path.join(source_path, f"embeddings/{args.model}/layer_{args.layer}/embedding_pickles_{args.dataset}/test")

    collection_test = {}
    for pickled_emb in os.listdir(emb_dir) :
        if not pickled_emb.startswith('.') :
            with open(os.path.join(emb_dir, pickled_emb), 'rb') as f :
                emb = pickle.load(f)
            collection_test.update(emb)

#converting to list of dictionnaries

#train : 

collection_train_list = []
for key in collection_train.keys() :
    for embeddings in collection_train[key] :
        collection_dict = {"label":key, "embedding":embeddings.flatten()}
        collection_train_list.append(collection_dict)

print("train size : ", len(collection_train_list))

ds_train = Dataset.from_list(collection_train_list)
ds_train = ds_train.class_encode_column('label')

#print(ds_train.features['label'].names)

#test :
if not args.dataset == "10_50" :
    collection_test_list = []
    for key in collection_train.keys() :
        for embeddings in collection_test[key] :
            collection_dict = {"label":key, "embedding":embeddings.flatten()}
            collection_test_list.append(collection_dict)

    print("test size : ", len(collection_test_list))

    ds_test = Dataset.from_list(collection_test_list)
    ds_test = ds_test.class_encode_column('label')

#print(ds_test.features['label'].names)

#making torch dataset and splitting val from train : 

if not args.dataset=="10_50" : 
    train_valid = ds_train.train_test_split(shuffle=True, test_size=0.2, stratify_by_column='label')
#test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    ds = DatasetDict(
        {
            "train": train_valid["train"],
            "test": ds_test,
            "val": train_valid["test"],
        }
    ).with_format("torch")

else :
    train_valid = ds_train.train_test_split(shuffle=True, test_size=0.2, stratify_by_column='label')
    test_valid = train_valid["test"].train_test_split(shuffle=True, test_size=0.5, stratify_by_column='label')
    ds = DatasetDict(
        {
            "train": train_valid["train"],
            "test": test_valid["train"],
            "val": test_valid["test"],
        }
    ).with_format("torch")

print(ds)

##########################################################################################
######################################PyTorch model#######################################
##########################################################################################

#load datasets

train_loader, test_loader, val_loader = DataLoader(ds['train'], batch_size=144), DataLoader(ds['test']), DataLoader(ds['val'], batch_size=36)

#extract embedding dim from ds
embedding_dim = ds['train'][0]['embedding'].size(dim=0)
print("EMBEDDING DIM = ", embedding_dim)
#extract num labels from (huggingface) dataset
num_labels = len(ds_train.features['label'].names)

print("num_labels : ", num_labels)

# Define Linear Regression Class
class LinearRegressionSimple(nn.Module):
    def __init__(self, in_features=embedding_dim, out_features=num_labels):
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features, 
            out_features=out_features
            )
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)
    
# Instantiating our Model
probe = LinearRegressionSimple()

# Adding Loss Function and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(params=probe.parameters(), lr=1e-3)
#optimizer = optim.Adam(params=probe.parameters(), lr=1e-3)

# Defining our device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting up lists to track losses
train_losses = []
val_losses = []


#model functions : 

def train(model, train_loader, criterion, optimizer, epoch, num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        #print(inputs)
        inputs = batch['embedding'].to(device)
        targets = batch['label'].to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = train_loss/len(train_loader)
    #print(f'Epoch [{epoch + 1:03}/{num_epochs:03}] | Train Loss: {avg_loss:.4f}')
    train_losses.append(train_loss/len(train_loader))

# Defining a Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader :
            inputs = batch['embedding'].to(device)
            targets = batch['label'].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    #print(f'Validation Loss: {avg_loss:.4f}')
    val_losses.append(avg_loss)

def test(model, test_loader, criterion, device) :
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader :
            for (targets, inputs) in zip(batch['label'], batch['embedding']):
                inputs=inputs.to(device)
                targets=targets.to(device)

def compute_metrics(eval_pred) :
    compute_accuracy_metric = evaluate.load("mount/Gibbon_ID_probing/evaluate_main/metrics/accuracy/accuracy.py")
    compute_f1_metric = evaluate.load("mount/Gibbon_ID_probing/evaluate_main/metrics/f1/f1.py")
    compute_matthews_metric = evaluate.load("mount/Gibbon_ID_probing/evaluate_main/metrics/matthews_correlation/matthews_correlation.py")
    logits, labels = eval_pred
    predictions = [np.argmax(logit.cpu(), axis=-1) for logit in logits]
    accuracy = compute_accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = compute_f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    matthews_correlation = compute_matthews_metric.compute(predictions=predictions, references=labels)
    return {"accuracy":accuracy, "f1":f1, "matthews_correlation":matthews_correlation}


#training loop :
print("TRAINING")
num_epochs = args.epochs
probe.to(device)
evaluations = []
for epoch in tqdm.tqdm(range(num_epochs)):
    train(probe, train_loader, criterion, optimizer, epoch, num_epochs)
    validate(probe, val_loader, criterion, device)



source_path = "/gpfswork/rech/jeb/uuz13rj/mount/Gibbon_ID_probing"

#plotting learning curves
plt.plot(range(len(train_losses)), train_losses, val_losses)
plt.savefig(os.path.join(source_path, f'results/LR/{args.model}_{args.layer}LC.png'))

print("EVALUATING")
#Evaluation :
eval_pred = ([],[])
for example in tqdm.tqdm(test_loader) :
    inputs = example['embedding'].to(device)
    eval_pred[0].append(torch.Tensor.detach(probe(inputs)))
    eval_pred[1].append(example['label'])

eval_dict = compute_metrics(eval_pred)

print(eval_dict)

eval_dict.update({'dataset' : args.dataset, 'model' : args.model, 'epochs' : args.epochs})


source_path = "mount/Gibbon_ID_probing"

if args.save_results : 
    with open(os.path.join(source_path, f"results/{args.model}_layer_{args.layer}_results.csv"), 'a', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(eval_dict.values())


#Confusion matrix : 

def compute_confusion_matrix(test_loader) :

    eval_pred = ([],[])
    for example in test_loader :
        inputs = example['embedding'].to(device)
        #eval_pred[0].append(torch.Tensor.detach(probe(inputs)))
        temp = probe(inputs).detach().cpu()
        eval_pred[0].append(temp.numpy())
        eval_pred[1].append(example['label'])

    y_true = [value.item() for value in eval_pred[1]]
    y_pred = [np.argmax(value[0], axis=-1) for value in eval_pred[0]]
    y_pred = [value.item() for value in y_pred]
    cf_matrix = confusion_matrix(y_true, y_pred)
    return cf_matrix

print(compute_confusion_matrix(test_loader))
