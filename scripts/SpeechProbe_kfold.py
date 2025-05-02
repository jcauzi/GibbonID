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
from sklearn.model_selection import KFold


##########################################################################################
####################################Arguments############################################# 
##########################################################################################


parser = argparse.ArgumentParser(description='Specify arguments Speech probe')
 
parser.add_argument('-d','--dataset', help='The dataset [all, 25_10, noise]',required=True)
parser.add_argument('-m','--model', help='The pre-trained model name [Unispeech, HuBERT, Vggish]',required=True)
parser.add_argument('-e','--epochs', help='number of epochs',required=True, type=int)
parser.add_argument('-r', '--save_results', action='store_true', help='Wether to save results as new line in results.csv')
parser.add_argument('-l', '--layer', help='The embedding layer to be probed', required=False)

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
ds_train = ds_train.with_format("torch")

#print(ds_train.features['label'].names)


#making torch dataset and splitting val from train : 


train_test = ds_train.train_test_split(shuffle=True, test_size=0.2, stratify_by_column='label')
ds = DatasetDict(
    {
        "train": train_test["train"],
        "test": train_test["test"],
    }
).with_format("torch")

print(ds)

##########################################################################################
######################################Training functions##################################
##########################################################################################


#training functions : 
def train(model, train_loader, criterion, optimizer, epoch, num_epochs, device):
    model.train()
    train_loss = 0.0

    for idx, data in enumerate(train_loader):
        inputs = data['embedding'].to(device)
        inputs = inputs.float()
        targets = data['label'].to(device)
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

##########################################################################################
###################################### k-fold #############################################
##########################################################################################


# Defining our device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 20
#extract embedding dim from ds
embedding_dim = len(ds_train[0]['embedding'])
#extract num labels from (huggingface) dataset
num_labels = len(ds_train.features['label'].names)

k_folds = 10

kf = KFold(n_splits=k_folds, shuffle=True)

accuracies, f1s, matthews = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(ds_train)):
    print(f"Fold {fold + 1}")
    print("-------")

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    test_loader = DataLoader(
        dataset=ds_train,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )


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
    probe.to(device)

    # Adding Loss Function and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(params=probe.parameters(), lr=1e-3)
    #optimizer = optim.Adam(params=probe.parameters(), lr=1e-3)

    # Setting up lists to track losses
    train_losses = []
 
    #training loop :
    print("TRAINING")
    num_epochs = args.epochs
    probe.to(device)
    evaluations = []

    print(train_loader)

    for epoch in tqdm.tqdm(range(num_epochs)):
        train(probe, train_loader, criterion, optimizer, epoch, num_epochs, device)

    source_path = "/gpfswork/rech/jeb/uuz13rj/mount/Gibbon_ID_probing"

    #plotting learning curves
    plt.plot(range(len(train_losses)), train_losses)
    plt.savefig(os.path.join(source_path, f'results/LR/{args.model}_{args.layer}LC.png'))


    #Evaluation :
    eval_pred = ([],[])
    for batch in enumerate(test_loader) :
        inputs = batch[1]['embedding'].to(device)
        inputs = inputs.float()
        eval_pred[0].append(torch.Tensor.detach(probe(inputs)))
        eval_pred[1].append(batch[1]['label'])

    eval_dict = compute_metrics(eval_pred)

    print(eval_dict)

    eval_dict.update({'dataset' : args.dataset, 'model' : args.model, "epochs" : args.epochs, "folds" : f"{fold + 1}"})

    accuracies.append(eval_dict['accuracy']['accuracy'])
    f1s.append(eval_dict['f1']['f1'])
    matthews.append(eval_dict['matthews_correlation']['matthews_correlation'])

eval_dict['accuracy'] = np.mean(accuracies)
eval_dict['f1'] = np.mean(f1s)
eval_dict['matthews_correlation'] = np.mean(matthews)

source_path = "mount/Gibbon_ID_probing"

print("final lap :", eval_dict)

if args.save_results : 
    print("saving results")
    with open(os.path.join(source_path, f"results/Speech_layer_{args.layer}_results_k_fold.csv"), 'a', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(eval_dict.values())
    print("saved")

#Confusion matrix : 

def compute_confusion_matrix(test_loader) :

    eval_pred = ([],[])
    for example in test_loader :
        eval_pred[0].append(torch.Tensor.detach(probe(example['embedding'])))
        eval_pred[1].append(example['label'])

    y_true = [value.item() for value in eval_pred[1]]
    y_pred = [np.argmax(value[0], axis=-1) for value in eval_pred[0]]
    y_pred = [value.item() for value in y_pred]
    cf_matrix = confusion_matrix(y_true, y_pred)
    return cf_matrix
