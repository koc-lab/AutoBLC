import codecs
import time
import os
import pandas as pd



file_name="mills_words.csv" 
hollinks= pd.read_csv(file_name)
source_path = "/auto/k2/aykut3/Basic_Level_NLP/word2gm-irem/wiki_data5.txt"
hollinks= list(hollinks["Synset"])
hollinks_filter=[]
for h in hollinks:
    if ('-' in h) or ('_' in h):
        h=h.replace('_'," ")
        #h=h.replace('-'," ")
        ind1= h.find("'")
        ind=h.find(".")
        h=h[ind1+1:ind]
        hollinks_filter.append(h)
    else:
       ind1= h.find("'")
       ind=h.find(".")
       h=h[ind1+1:ind]
       hollinks_filter.append(h)
       
       
       
       
       


import mmap

with open(source_path, 'r') as f:
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
file2list=file2list.decode("utf-8") 
file2list= file2list.replace('\n',' ')
with open("mills_text.txt", "w") as fwrite: 
      print("Writing started")
      start_time=time.time()
      for word in hollinks_filter:
         if word in file2list:
            ind=file2list.find(word)
            word2extract= file2list[ind-100:ind+100]
            fwrite.write(word2extract+'\n')
print("Reading finished")
end_time=time.time()
print("Writing and creating text last", end_time-start_time, " second")
fwrite.close()