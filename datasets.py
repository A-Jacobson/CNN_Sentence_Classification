import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SubjObjDataset(Dataset):

    def __init__(self, path, vectorizer, tokenizer=None):
        self.corpus = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.class2idx = {cls: idx for idx, cls in enumerate(sorted(np.unique(self.corpus['labels'])))}
        self._tokenize_corpus()
        self._vectorize_corpus()

    def _tokenize_corpus(self):
        if self.tokenizer:
            self.corpus['tokens'] = self.corpus['sentences'].apply(self.tokenizer)
        else:
            self.corpus['tokens'] = self.corpus['sentences'].apply(lambda x: x.split())

    def _vectorize_corpus(self):
        if not self.vectorizer.vocabulary:
            self.vectorizer.fit(self.corpus['tokens'])
        self.corpus['vectors'] = self.corpus['tokens'].apply(self.vectorizer.transform_document)

    def __getitem__(self, index):
        sentence = self.corpus['vectors'].iloc[index]
        target = [self.class2idx[self.corpus['labels'].iloc[index]]]
        return torch.LongTensor(sentence), torch.LongTensor(target)

    def __len__(self):
        return len(self.corpus)

