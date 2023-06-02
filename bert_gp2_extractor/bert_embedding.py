import torch
from transformers import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import mmap
import codecs
import time
import os
import pandas as pd
import numpy as np
# Load pre-trained model tokenizer (vocabulary)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


hollinks= pd.read_csv("mills_words.csv")  # hollink_label.csv --> for hollink dataset
hollinks= list(hollinks["Synset"]) # list of words that you want to have embeddings

hollinks_filter=[]
for h in hollinks:
  hollinks_filter.append(h[2:-1])

# Read the txt file which you extract embeddings
#bert_hollink_text.txt or bert_mills_text.txt
with open("mills_text.txt", 'r') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # process the memory-mapped file here
        print("Reading started")
        file2list=mm.read()
        start_time=time.time()
        print("Reading finished")
        end_time=time.time()

mm.close()
f.close()

print("Reading last", end_time-start_time, " second")
print("Length of the file as string is ", len(file2list))     
#print(file2list)




file2list=file2list.decode("utf-8") 
file2list= file2list.replace('\n',' ')
# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(file2list)

tokenized_text.append("[UNK]")
tokenized_text.append("[UNK]")


tokenized_text_reshaped=np.reshape(tokenized_text,(751,50))
#If you want to use CLS and SEP for each batch
'''
cls_tokens= np.expand_dims(np.array(["[CLS]"]*470,dtype= tokenized_text_reshaped.dtype),axis=1)
sep_tokens= np.expand_dims(np.array(["[SEP]"]*470,dtype= tokenized_text_reshaped.dtype),axis=1)

print(tokenized_text_reshaped.shape)
print(cls_tokens.shape)
print(sep_tokens.shape)
tokenized_text_with_cls_sep= np.concatenate((cls_tokens,tokenized_text_reshaped,sep_tokens), axis=1)
tokenized_text=np.reshape(tokenized_text_with_cls_sep,(470*79,)).tolist()
tokenized_text_reshaped = np.reshape(tokenized_text,(470,79))
'''




# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Arrange dimensions according to your token size. Reshape the total number of tokens as BxN--> B: batch size, N: sequence length

tokens_tensor= torch.reshape(tokens_tensor,(751,50))
segments_tensors= torch.reshape(segments_tensors,(751,50))
print(tokens_tensor.shape)
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers. 
with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    hidden_states = outputs[2]


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())


# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)


print(token_embeddings.size())

## Comment out according your choice, last hidden state or average of final hidden state

mean_token_vec = torch.mean(token_embeddings[-4:], dim=0)
#mean_token_vec = token_embeddings[-1]
print(mean_token_vec.size())


# Here, find the token index and corresponding embedding in 2D list [MxN]--> M:sample size, N: embedding size

def index_2d(myList, v):
    checkme=False
    for i, x in enumerate(myList):
        if v in x:
            checkme=True
            return [i, x.index(v)]
    if checkme==False:
       return [len(myList)-1,len(myList[0])-1]

hollink_embedding= np.zeros((len(hollinks_filter),mean_token_vec.size(dim=2)))

for i,w in enumerate(hollinks_filter):
  tokenized_word= tokenizer.tokenize(w)
  vec_sum = 0
  for tw in tokenized_word:
      ind =  index_2d(tokenized_text_reshaped.tolist(), tw)
      vec_sum= vec_sum+mean_token_vec[ind[0],ind[1],:]
  hollink_embedding[i,:]=vec_sum/len(tokenized_word)   

cols = []

for i in range(len(hollink_embedding[0])):
   cols.append("dim_"+str(i))
df= pd.DataFrame(hollink_embedding,columns=cols)
df.to_csv("bert_mills.csv",index=False) #gpt2_hollink.csv --> for hollink

