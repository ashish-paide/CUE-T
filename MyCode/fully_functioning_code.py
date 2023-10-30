# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:21:56.220067Z","iopub.execute_input":"2023-05-03T19:21:56.220941Z","iopub.status.idle":"2023-05-03T19:21:56.239108Z","shell.execute_reply.started":"2023-05-03T19:21:56.220901Z","shell.execute_reply":"2023-05-03T19:21:56.237847Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # Libraries

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:21:56.241245Z","iopub.execute_input":"2023-05-03T19:21:56.241642Z","iopub.status.idle":"2023-05-03T19:22:17.981000Z","shell.execute_reply.started":"2023-05-03T19:21:56.241602Z","shell.execute_reply":"2023-05-03T19:22:17.979436Z"}}
!pip install transformers
!pip install torch

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:22:17.985132Z","iopub.execute_input":"2023-05-03T19:22:17.985581Z","iopub.status.idle":"2023-05-03T19:22:28.941650Z","shell.execute_reply.started":"2023-05-03T19:22:17.985537Z","shell.execute_reply":"2023-05-03T19:22:28.940560Z"}}
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from torch import nn
from transformers import DistilBertModel
import pickle
import pandas as pd

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:22:28.944626Z","iopub.execute_input":"2023-05-03T19:22:28.945403Z","iopub.status.idle":"2023-05-03T19:22:28.950285Z","shell.execute_reply.started":"2023-05-03T19:22:28.945366Z","shell.execute_reply":"2023-05-03T19:22:28.948710Z"}}
PREFIX="/kaggle/input/sarcasm-detection/"
tweets_file="sanitised_fixed_len.csv"
user_tweets_file="user_tweets_latest"

# %% [code] {"execution":{"iopub.status.busy":"2023-05-03T19:23:04.091284Z","iopub.execute_input":"2023-05-03T19:23:04.092384Z","iopub.status.idle":"2023-05-03T19:23:04.097065Z","shell.execute_reply.started":"2023-05-03T19:23:04.092330Z","shell.execute_reply":"2023-05-03T19:23:04.095272Z"}}
import os
os.environ["WANDB_DISABLED"] = "true"

# %% [markdown]
# # Train val test split

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:04.718019Z","iopub.execute_input":"2023-05-03T19:23:04.718686Z","iopub.status.idle":"2023-05-03T19:23:04.761995Z","shell.execute_reply.started":"2023-05-03T19:23:04.718643Z","shell.execute_reply":"2023-05-03T19:23:04.760967Z"}}
df=pd.read_csv(PREFIX+tweets_file)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:04.922737Z","iopub.execute_input":"2023-05-03T19:23:04.923084Z","iopub.status.idle":"2023-05-03T19:23:04.942330Z","shell.execute_reply.started":"2023-05-03T19:23:04.923050Z","shell.execute_reply":"2023-05-03T19:23:04.941211Z"}}
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:05.113926Z","iopub.execute_input":"2023-05-03T19:23:05.114657Z","iopub.status.idle":"2023-05-03T19:23:05.123221Z","shell.execute_reply.started":"2023-05-03T19:23:05.114609Z","shell.execute_reply":"2023-05-03T19:23:05.121380Z"}}
df.iloc[4466,:]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:05.307048Z","iopub.execute_input":"2023-05-03T19:23:05.307742Z","iopub.status.idle":"2023-05-03T19:23:05.313905Z","shell.execute_reply.started":"2023-05-03T19:23:05.307703Z","shell.execute_reply":"2023-05-03T19:23:05.312803Z"}}
def load_and_preprocess_data():
    df=pd.read_csv(PREFIX+tweets_file)
    tweets=df['tweet_text']
    tweets=list(tweets)
    #print(tweets)
    s_score=df["sarcasm_score"]
    a_names=df['author_full_name']
    return (tweets,s_score,a_names)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:05.471399Z","iopub.execute_input":"2023-05-03T19:23:05.472065Z","iopub.status.idle":"2023-05-03T19:23:05.506580Z","shell.execute_reply.started":"2023-05-03T19:23:05.472025Z","shell.execute_reply":"2023-05-03T19:23:05.505620Z"}}
# Replace this with your data loading and preprocessing
texts, labels,names = load_and_preprocess_data()

train_texts, val_texts, train_labels, val_labels,train_names,val_names= train_test_split(texts, labels,names, test_size=0.2, random_state=42)
val_texts,test_texts,val_labels,test_labels,val_names,test_names=train_test_split(val_texts,val_labels,val_names,test_size=0.5,random_state=42)
train_labels = pd.Series(train_labels).reset_index(drop=True)
val_labels = pd.Series(val_labels).reset_index(drop=True)
test_labels = pd.Series(test_labels).reset_index(drop=True)

train_names = pd.Series(train_names).reset_index(drop=True)
val_names = pd.Series(val_names).reset_index(drop=True)
test_names = pd.Series(test_names).reset_index(drop=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:05.660211Z","iopub.execute_input":"2023-05-03T19:23:05.660907Z","iopub.status.idle":"2023-05-03T19:23:05.668583Z","shell.execute_reply.started":"2023-05-03T19:23:05.660857Z","shell.execute_reply":"2023-05-03T19:23:05.667390Z"}}
train_texts[2756]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:05.841572Z","iopub.execute_input":"2023-05-03T19:23:05.842554Z","iopub.status.idle":"2023-05-03T19:23:05.850897Z","shell.execute_reply.started":"2023-05-03T19:23:05.842504Z","shell.execute_reply":"2023-05-03T19:23:05.849649Z"}}
train_labels[2756]

# %% [markdown]
# # Tokenization

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:06.220408Z","iopub.execute_input":"2023-05-03T19:23:06.221116Z","iopub.status.idle":"2023-05-03T19:23:11.556174Z","shell.execute_reply.started":"2023-05-03T19:23:06.221075Z","shell.execute_reply":"2023-05-03T19:23:11.555177Z"}}
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# %% [markdown]
# ## Maximum Number of tokens

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:11.558231Z","iopub.execute_input":"2023-05-03T19:23:11.559213Z","iopub.status.idle":"2023-05-03T19:23:12.371262Z","shell.execute_reply.started":"2023-05-03T19:23:11.559165Z","shell.execute_reply":"2023-05-03T19:23:12.370112Z"}}
# Tokenize the tweets and calculate the lengths
tokenized_tweets = [tokenizer.encode(tweet) for tweet in texts]
lengths = [len(tweet) for tweet in tokenized_tweets]
max_length=max(lengths)
print("Max length= ",max_length)

# %% [markdown]
# ## Encodings

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:12.372837Z","iopub.execute_input":"2023-05-03T19:23:12.373532Z","iopub.status.idle":"2023-05-03T19:23:12.958580Z","shell.execute_reply.started":"2023-05-03T19:23:12.373489Z","shell.execute_reply":"2023-05-03T19:23:12.957503Z"}}
max_length=max_length+10
train_encodings = tokenizer(train_texts, return_tensors='pt', padding='max_length', max_length=max_length, truncation=False)
val_encodings= tokenizer(val_texts,return_tensors='pt',padding='max_length',max_length=max_length,truncation=False)
test_encodings= tokenizer(test_texts,return_tensors='pt',padding='max_length',max_length=max_length,truncation=False)

# %% [markdown]
# # User Representation

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:12.961315Z","iopub.execute_input":"2023-05-03T19:23:12.961609Z","iopub.status.idle":"2023-05-03T19:23:13.554880Z","shell.execute_reply.started":"2023-05-03T19:23:12.961579Z","shell.execute_reply":"2023-05-03T19:23:13.553712Z"}}
u_file="user2vec_100_master.txt"
names=[]
with open(PREFIX+u_file,"r") as f:
    lines=f.readlines()
    for idx,line in enumerate(lines):
        if (idx==0):
            continue
        vector_list=[]
        for it,item in enumerate(line.split(" ")):
            if (it==0):
                name=item
                names.append(name)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:13.559709Z","iopub.execute_input":"2023-05-03T19:23:13.560627Z","iopub.status.idle":"2023-05-03T19:23:13.570394Z","shell.execute_reply.started":"2023-05-03T19:23:13.560582Z","shell.execute_reply":"2023-05-03T19:23:13.569087Z"}}
print(len(names))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:13.572228Z","iopub.execute_input":"2023-05-03T19:23:13.573011Z","iopub.status.idle":"2023-05-03T19:23:13.583805Z","shell.execute_reply.started":"2023-05-03T19:23:13.572970Z","shell.execute_reply":"2023-05-03T19:23:13.582541Z"}}
user_representations1={name:[] for name in names}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:13.585568Z","iopub.execute_input":"2023-05-03T19:23:13.586379Z","iopub.status.idle":"2023-05-03T19:23:13.593406Z","shell.execute_reply.started":"2023-05-03T19:23:13.586338Z","shell.execute_reply":"2023-05-03T19:23:13.592111Z"}}
print(len(user_representations1))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:13.595096Z","iopub.execute_input":"2023-05-03T19:23:13.596000Z","iopub.status.idle":"2023-05-03T19:23:13.602805Z","shell.execute_reply.started":"2023-05-03T19:23:13.595961Z","shell.execute_reply":"2023-05-03T19:23:13.601526Z"}}
#user_representations[train_names.iloc[4466]]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:13.604595Z","iopub.execute_input":"2023-05-03T19:23:13.605310Z","iopub.status.idle":"2023-05-03T19:23:14.499874Z","shell.execute_reply.started":"2023-05-03T19:23:13.605271Z","shell.execute_reply":"2023-05-03T19:23:14.498772Z"}}
with open(PREFIX+u_file,"r") as f:
    lines=f.readlines()
    for idx, line in enumerate(lines):
        if(idx==0):
            continue
        vector_list=[]
        for it,item in enumerate(line.split(" ")):
            if (it==0):
                name=item
            else:
                vector_list.append(float(item))
        user_representations1[name]=vector_list

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:14.504686Z","iopub.execute_input":"2023-05-03T19:23:14.505054Z","iopub.status.idle":"2023-05-03T19:23:14.511256Z","shell.execute_reply.started":"2023-05-03T19:23:14.505013Z","shell.execute_reply":"2023-05-03T19:23:14.509969Z"}}
print(len(user_representations1['gansome']))

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-05-03T19:23:14.512829Z","iopub.execute_input":"2023-05-03T19:23:14.513652Z","iopub.status.idle":"2023-05-03T19:23:14.603490Z","shell.execute_reply.started":"2023-05-03T19:23:14.513614Z","shell.execute_reply":"2023-05-03T19:23:14.602207Z"}}
created_user_file="/kaggle/input/sarcasm-detection/user_emb_vec100_Doc2Vec"
f=open(created_user_file,'rb')
user_representations2=pickle.load(f)
# print(type(d))
# print(d['DamonLThomas'])
# print(len(d['DamonLThomas']))

# %% [code] {"execution":{"iopub.status.busy":"2023-05-03T19:23:14.604903Z","iopub.execute_input":"2023-05-03T19:23:14.605966Z","iopub.status.idle":"2023-05-03T19:23:14.611317Z","shell.execute_reply.started":"2023-05-03T19:23:14.605922Z","shell.execute_reply":"2023-05-03T19:23:14.610183Z"}}
# choice=int(input("Choose user representation: "))
# if (choice==1):
#     user_representations=user_representations1
# elif (choice==2):
#     user_representations=user_representations2
# else:
#     print("Wrong input!")

# %% [code] {"execution":{"iopub.status.busy":"2023-05-03T19:23:14.613006Z","iopub.execute_input":"2023-05-03T19:23:14.613873Z","iopub.status.idle":"2023-05-03T19:23:14.619970Z","shell.execute_reply.started":"2023-05-03T19:23:14.613834Z","shell.execute_reply":"2023-05-03T19:23:14.618907Z"}}
#There is better
user_representations=user_representations1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:14.622504Z","iopub.execute_input":"2023-05-03T19:23:14.623208Z","iopub.status.idle":"2023-05-03T19:23:14.631466Z","shell.execute_reply.started":"2023-05-03T19:23:14.623168Z","shell.execute_reply":"2023-05-03T19:23:14.630500Z"}}
class CustomDistilBertWithUserRepresentation(nn.Module):
    def __init__(self, user_r_dim, n_classes):
        super(CustomDistilBertWithUserRepresentation, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.user_representation_dim = user_r_dim
        hidden_size=self.distilbert.config.hidden_size 
        self.classifier = nn.Linear(hidden_size+ user_r_dim, n_classes)

    def forward(self, input_ids, attention_mask, user_representation):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output.last_hidden_state
        #print("Hidden state shape= ",hidden_state.shape)
        
        mean_pooled_output = torch.mean(hidden_state, dim=1)
        #print("Mean pooled shape= ",mean_pooled_output.shape)
        
        merged_output = torch.cat((mean_pooled_output, user_representation), dim=1)
        logits = self.classifier(merged_output)

        return logits

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:14.633071Z","iopub.execute_input":"2023-05-03T19:23:14.633481Z","iopub.status.idle":"2023-05-03T19:23:14.698683Z","shell.execute_reply.started":"2023-05-03T19:23:14.633405Z","shell.execute_reply":"2023-05-03T19:23:14.697277Z"}}
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, author_names, user_representations,device):
        self.encodings = encodings
        self.labels = labels
        self.author_names = author_names
        self.user_representations = user_representations
        self.device=device

    def __getitem__(self, idx):
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        author_name = self.author_names[idx]
        default_representation = torch.as_tensor(torch.zeros_like(torch.as_tensor(next(iter(self.user_representations.values()))))).to(self.device)
        user_representation = self.user_representations.get(author_name, default_representation)
        item['user_representation'] = torch.as_tensor(user_representation).to(self.device)
        item['labels'] = torch.as_tensor(int(self.labels[idx])).to(self.device)  # Make sure to include the 'labels' key
        return item

    def __len__(self):
        return len(self.labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = SarcasmDataset(train_encodings, train_labels, train_names, user_representations,device)
val_dataset = SarcasmDataset(val_encodings, val_labels, val_names, user_representations,device)
test_dataset=SarcasmDataset(test_encodings,test_labels,test_names,user_representations,device)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:14.701119Z","iopub.execute_input":"2023-05-03T19:23:14.702110Z","iopub.status.idle":"2023-05-03T19:23:16.697120Z","shell.execute_reply.started":"2023-05-03T19:23:14.702068Z","shell.execute_reply":"2023-05-03T19:23:16.696032Z"}}
model = CustomDistilBertWithUserRepresentation(user_r_dim=100, n_classes=2)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:16.698724Z","iopub.execute_input":"2023-05-03T19:23:16.699829Z","iopub.status.idle":"2023-05-03T19:23:16.708498Z","shell.execute_reply.started":"2023-05-03T19:23:16.699782Z","shell.execute_reply":"2023-05-03T19:23:16.707284Z"}}
from datasets import load_metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")

    acc_score = accuracy.compute(predictions=preds, references=labels)
    prec_score = precision.compute(predictions=preds, references=labels, average="binary")
    rec_score = recall.compute(predictions=preds, references=labels, average="binary")
    f1_score = f1.compute(predictions=preds, references=labels, average="binary")
    acc_score=acc_score['accuracy']
    prec_score=prec_score['precision']
    rec_score=rec_score['recall']
    f1_score=f1_score['f1']
    return {
        "accuracy": acc_score,
        "precision": prec_score,
        "recall": rec_score,
        "f1": f1_score,
    }

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:16.710329Z","iopub.execute_input":"2023-05-03T19:23:16.710747Z","iopub.status.idle":"2023-05-03T19:23:16.736646Z","shell.execute_reply.started":"2023-05-03T19:23:16.710680Z","shell.execute_reply":"2023-05-03T19:23:16.735694Z"}}
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=5,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    learning_rate=1e-5,
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,  # log/save every x steps
    evaluation_strategy="epoch",  # evaluation is done at the end of each epoch
    save_strategy="epoch",  # checkpoint saving is done at the end of each epoch
    save_total_limit=2,  # maximum number of checkpoints to keep
    load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
    metric_for_best_model="f1",  # choose the metric for the best model
    greater_is_better=True,  # specify whether a higher or lower metric value is better
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:16.738268Z","iopub.execute_input":"2023-05-03T19:23:16.738649Z","iopub.status.idle":"2023-05-03T19:23:20.951377Z","shell.execute_reply.started":"2023-05-03T19:23:16.738600Z","shell.execute_reply":"2023-05-03T19:23:20.950312Z"}}
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        
    def training_step(self, model, inputs):
        model.train()
        #print(inputs.keys())
        
        labels = inputs.pop("labels")
        user_representation = inputs.pop("user_representation")
        inputs = self._prepare_inputs(inputs)
        loss, outputs = self.compute_loss(model, inputs, labels, user_representation, return_outputs=True)
        loss.backward()  # Compute gradients using the backward method
        return loss

    def compute_loss(self, model, inputs, labels, user_representation, return_outputs=False):
        outputs = model(**inputs, user_representation=user_representation)
        logits = outputs.float()
        loss = nn.CrossEntropyLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset

        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            shuffle=True,
        )
    
    def get_eval_dataloader(self, dataset=None, remove_unused_columns=True, repeat=False, seed= 42):
        if dataset is None:
            dataset = self.eval_dataset

        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            shuffle=False,
        )
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if ("labels" in inputs.keys()):
            has_labels=True
            extracted_labels=inputs.pop("labels")
        inputs = self._prepare_inputs(inputs)

        # Get user representation from inputs and remove it before forwarding to the model
        user_representation = inputs.pop("user_representation")

        with torch.no_grad():
            if has_labels:
                if prediction_loss_only:
                    loss = self.compute_loss(model, inputs, extracted_labels, user_representation)
                    return (loss, None, None)
                else:
                    loss, outputs = self.compute_loss(model, inputs, extracted_labels, user_representation, return_outputs=True)
            else:
                loss = None
                outputs = model(**inputs, user_representation=user_representation)
            logits = outputs.float() if not self.args.prediction_loss_only else None

        if self.args.prediction_loss_only:
            return (loss, None, None)
        elif has_labels:
            labels = extracted_labels
            return (loss, logits, labels)
        else:
            return (loss, logits, None)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:23:20.952908Z","iopub.execute_input":"2023-05-03T19:23:20.953415Z","iopub.status.idle":"2023-05-03T19:24:00.950521Z","shell.execute_reply.started":"2023-05-03T19:23:20.953370Z","shell.execute_reply":"2023-05-03T19:24:00.948407Z"}}
trainer.train()
trainer.evaluate()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-03T19:24:00.951651Z","iopub.status.idle":"2023-05-03T19:24:00.952160Z","shell.execute_reply.started":"2023-05-03T19:24:00.951882Z","shell.execute_reply":"2023-05-03T19:24:00.951908Z"}}
# Evaluate the model on the test dataset
test_results = trainer.evaluate(test_dataset)
print("Accuracy: ",test_results['eval_accuracy'])
print("Precision: ",test_results['eval_precision'])
print("Recall: ",test_results['eval_recall'])
print("F1 score: ",test_results['eval_f1'])

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code]

