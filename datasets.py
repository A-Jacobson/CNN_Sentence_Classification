import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SubjObjDataset(Dataset):

    def __init__(self, path, vectorizer, tokenizer=None, stopwords=None):
        self.corpus = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.stopwords = stopwords
        self.class2idx = {cls: idx for idx, cls in enumerate(sorted(np.unique(self.corpus['labels'])))}
        self._tokenize_corpus()
        if self.stopwords:
            self._remove_stopwords()
        self._vectorize_corpus()

    def _remove_stopwords(self):
        stopfilter = lambda doc: [word for word in doc if word not in self.stopwords]
        self.corpus['tokens'] = self.corpus['tokens'].apply(stopfilter)

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

