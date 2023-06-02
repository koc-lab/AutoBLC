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
from transformers import GPT2Tokenizer, GPT2Model
# Load pre-trained mod

filename="mills_words.csv"  # hollink_label.csv --> for hollink dataset
hollinks= pd.read_csv(filename) # list of words that you want to have embeddings
hollinks= list(hollinks["Synset"])

hollinks_filter=[]
for h in hollinks:
  hollinks_filter.append(h[2:-1])


# Read the txt file which you extract embeddings
with open("mills_text.txt", 'r') as f: # hollink_text.txt --> for hollink_dataset
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


# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
tokenized_text= tokenizer.encode(file2list)


# Arrange dimensions according to your token size. Reshape the total number of tokens as BxN--> B: batch size, N: sequence length
input_ids = torch.tensor(tokenized_text)
print("Length of the input ids",len(input_ids))

input_ids_reshaped=torch.reshape(input_ids,(48,797))

print(input_ids_reshaped.shape)



# Get hidden states from model
with torch.no_grad():
    outputs= model(input_ids_reshaped,output_hidden_states=True)[2]
    hidden_states=outputs
# Extract word embeddings from last layer of hidden states

print(hidden_states[0].shape)


# Stack all hidden layers and take the average of last four layers for more global representation
token_embeddings = torch.stack(hidden_states, dim=0)
mean_word_embeddings = torch.mean(token_embeddings[-4:],dim=0)
print(mean_word_embeddings.shape)


# Here, find the token index and corresponding embedding in 2D list [MxN]--> M:sample size, N: embedding size
def index_2d(myList, v):
    checkme=False
    for i, x in enumerate(myList):
        if v in x:
            checkme=True
            return [i, x.index(v)]
    if checkme==False:
       return [len(myList)-1,len(myList[0])-1]

hollink_embedding= np.zeros((len(hollinks_filter),mean_word_embeddings.size(dim=2)))


for i,w in enumerate(hollinks_filter):
  tokenized_word= tokenizer.encode(w)
  vec_sum = 0
  for tw in tokenized_word:
      ind =  index_2d(input_ids_reshaped.tolist(), tw)
      vec_sum= vec_sum+mean_word_embeddings[ind[0],ind[1],:]
  hollink_embedding[i,:]=vec_sum/len(tokenized_word)   




cols = []

for i in range(len(hollink_embedding[0])):
   cols.append("dim_"+str(i))
df= pd.DataFrame(hollink_embedding,columns=cols)
df.to_csv("gpt2_mills.csv",index=False)  #gpt2_hollink.csv --> for hollink
