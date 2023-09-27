# AutoBLC: 


This is the GitHub repository for the paper: E.Koc, İ.Şanlı A. Koç, **“ Word Embedding Based Basic Level Detection ”** submitted to IEEE Signal Processing Letters, 2023. In this study, we aim to automatically identify basic level categories using classical machine learning methods where the features are extracted using word embeddings including W2G, GPT-2 and BERT.

#### Extract Features from BERT and GPT-2 ####

To extract features from the BERT and GPT-2, go `bert_gpt2_extractor.py` and run scripts `bert_embedding.py` and `gpt2_embedding.py` using your corpus. In this folder, we also provide two small corpus for you to get familiar with the code.


#### Feature Selection ####

We share the **raw and selected features** in this repo. You can use the `feature_selection.ipynb` code to select your features based on the correlation between features and targets

#### Models ####

Using the `models.ipynb`, you can run the code for all models. We also provide the trained models for both datasets.

![GitHub Logo](https://github.com/koc-lab/AutoBLC/blob/main/blc_architecture.png)



