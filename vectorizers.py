from collections import Counter


class IndexVectorizer:
    """
    Transforms a Corpus into lists of word indices.
    """
    def __init__(self, max_words=None, min_frequency=None, start_end_tokens=False, maxlen=None):
        self.vocabulary = None
        self.vocabulary_size = 0
        self.word2idx = dict()
        self.idx2word = dict()
        self.max_words = max_words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        self.maxlen = maxlen

    def _find_max_document_length(self, corpus):
        self.maxlen = max(len(document) for document in corpus)
        if self.start_end_tokens:
            self.maxlen += 2

    def _build_vocabulary(self, corpus):
        vocabulary = Counter(word for document in corpus for word in document)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary) + 2  # padding and unk tokens
        if self.start_end_tokens:
            self.vocabulary_size += 2

    def _build_word_index(self):
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1

        if self.start_end_tokens:
            self.word2idx['<START>'] = 2
            self.word2idx['<END>'] = 3

        self.word2idx = {word: idx + len(self.word2idx) for idx,
                         word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def fit(self, corpus):
        if not self.maxlen:
            self._find_max_document_length(corpus)
        self._build_vocabulary(corpus)
        self._build_word_index()

    def pad_document_vector(self, vector):
        padding = self.maxlen - len(vector)
        vector.extend([1] * padding)
        return vector

    def add_start_end(self, vector):
        vector.append(3)
        return [2] + vector

    def transform_document(self, document):
        """
        Vectorize a single document
        """
        vector = [self.word2idx.get(word, 0) for word in document]
        if len(vector) > self.maxlen:
            return vector[:self.maxlen]
        if self.start_end_tokens:
            vector = self.add_start_end(vector)
        return self.pad_document_vector(vector)

    def transform(self, corpus):
        """
        Vectorizes a corpus in the form of a list of lists.
        A corpus is a list of documents and a document is a list of words.
        """
        return [self.transform_document(document) for document in corpus]
