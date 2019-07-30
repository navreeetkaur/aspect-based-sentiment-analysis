import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

from seqeval.metrics import f1_score

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

MAX_LEN = 512 #75
bs = 32

tags_vals = list(["O", "I", "B"])
tag2idx = {t: i for i, t in enumerate(tags_vals)}

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

def generate_corpus(filename):
    corpus = []
    soup = BeautifulSoup(open(filename, "r"), "lxml")
    idx = 0
    full_labels = []
    for r in soup.find_all('review'):
#         if idx>=1:
#             break
        doc = ""
        labels = []
        num_chars=0
        for idx, sentence in enumerate(r.find_all('sentence')):
            sent = sentence.text.strip()
#             print(sent)
            try:
                from_num = int(sentence.opinion.get("from"))
                to_num = int(sentence.opinion.get("to"))
            except:
                from_num=0
                to_num=0
            labels.append([num_chars+from_num+1,num_chars+to_num+1])
            doc += " " + sent
            num_chars = num_chars + (len(sent))+1
        k=0
        j=0
        for i in range(len(doc)):
            if k>=len(labels):
                break
            l = labels[k]
            if doc[i]==" ":
                if i==l[1]:
                    labels[k][1]=labels[k][1]-j
#                     print(i, doc[:i+1], "   ",l,"  ",j)
                    k+=1   
                j+=1
            if i==l[0]:
                labels[k][0]=labels[k][0]-j
#                 print(i, doc[:i+1], "   ",l,"  ",j)
            if doc[i]!=" " and i==l[1]:
                labels[k][1]=labels[k][1]-j
#                 print(i, doc[:i+1], "   ",l,"  ",j)
                k+=1
                
#             if l[0]!=l[1]:
#                 for i in range(len(sent)):
#                     if sent[i]==" ":
#                         j+=1
#                     if i==from_num:
#                         from_num=from_num-j
#                     if i==to_num:
#                         to_num=to_num-j
        corpus.append(doc)
        full_labels.append(labels)
    return corpus, full_labels


corpus, char_labels = generate_corpus("../data/official_data/EN_REST_SB1_TEST.xml.gold")
tokenized_texts = [tokenizer.tokenize(sent) for sent in corpus]


labels = []
for idx, text in enumerate(tokenized_texts):
    curr_labels = []
    k=0
    i=0
    for curr_idx, word in enumerate(text):           
        for char in word:
            if char!="#":
                if k>=len(char_labels[idx]):
                    break
                if i>char_labels[idx][k][1]:
                    k+=1
                    if k>=len(char_labels[idx]):
                        break
                if char_labels[idx][k][0]!=char_labels[idx][k][1]:
                    if i == char_labels[idx][k][0]:
                        curr_labels.append("B")
                    elif (char_labels[idx][k][0]<i and i<=char_labels[idx][k][1]-1):
                        if len(curr_labels)!=curr_idx+1:
                            curr_labels.append("I")
                i+=1
        if len(curr_labels)!=curr_idx+1:
            curr_labels.append("O")
    if idx<2:
        print(text)
        print(curr_labels)
        print()
    labels.append(curr_labels)

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
                                             tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

epochs = 5
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in tqdm_notebook(enumerate(train_dataloader)):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        print(loss)
        print("BPing. . ")
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
#         logits = logits.numpy()
        label_ids = b_labels.to('cpu').numpy()
#         label_ids = b_labels.numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))