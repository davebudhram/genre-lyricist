import nltk
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
SENTENCE_BEGIN = '<s>'
SENTENCE_END = '</s>'
NEWLINE_BREAK = 'newlinebreak'
NGRAM = 4


def tokenize_song(line, ngram=NGRAM):
  '''
  Tokenizes a song line by line. Returns a list of lists of tokens. Pads each line
  with ngram * SENTENCE_BEGIN and ngram * SENTENCE_END tokens to indicate the beginning and end of
  a sentence. Adds genre tag to each token.
  '''
  result = []
  sentences = line.split('\r\n')
  for sentence in sentences:
    sentence = sentence.lower()
    begining = [SENTENCE_BEGIN] * (ngram-2)
    end = [SENTENCE_END] * (ngram-2)
    tokens = begining + nltk.word_tokenize(sentence) + end
    result.append(tokens)
  return result


def tokenize_song_by_stanza(song, ngram=NGRAM):
  '''
  Tokenizes a song stanza by stanza. Returns a list of lists of tokens. Pads each line
  with ngram * SENTENCE_BEGIN and ngram * SENTENCE_END tokens to indicate the beginning and end of
  a sentence. Adds genre tag to each token. Also replaces each line break with a special line break token
  '''
  song = re.sub(r'\r\n', NEWLINE_BREAK + ' ', song)
  result = []
  sentences = song.split('\r\n  \r\n')
  for sentence in sentences:
    sentence = sentence.lower()
    begining = [SENTENCE_BEGIN] * (ngram-2)
    end = [SENTENCE_END] * (ngram-2)
    tokens = begining + nltk.word_tokenize(sentence) + end
    result.append(tokens)
  return result


def convertSamplesToEmbeddings(samples: list, index_to_embedding: dict):
    """
    Converts a list of samples to a list of embeddings.
    """
    embeddings = []
    for sample in samples:
        embedding = []
        for word in sample:
            embedding.append(index_to_embedding[word])
        embeddings.append(embedding)
    return np.array(embeddings)

def read_embeddings(filename: str, tokenizer: Tokenizer) -> dict:
    '''Loads and parses embeddings trained in earlier.
    Parameters:
        filename (str): path to file
        Tokenizer: tokenizer used to tokenize the data (needed to get the word to index mapping)
    Returns:
        (dict): mapping from index to its embedding vector
    '''
    # YOUR CODE HERE
    index_to_embedding = {}  # Mapping from index to its embedding vector
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.split()
            # Skip the first line of file
            if len(split_line) == 2:
                continue
            word = split_line[0]
            vector = [float(x) for x in split_line[1:]]
        
            if word in tokenizer.word_index:
                index_to_embedding[tokenizer.word_index[word]] = vector # Mapping from index to its embedding vector
    return index_to_embedding

def generate_ngram_training_samples(encoded: list, ngram: int):
    """
    Generates n-gram training samples from a list of encoded words. 
    """
    X, y = [], []
    ngram = ngram - 2
    for lyric in encoded:
      for i in range(1, len(lyric) - ngram):
          X.append([lyric[0]] + lyric[i:i + ngram])
          y.append(lyric[i + ngram])
    return X, y

# Function to generate batches of data
def data_generator(data, labels, index_to_embedding, batch_size, sequence_length, epochs):
    for epoch in range(epochs):
        num_batches = len(data) // batch_size
        while True:
            for i in range(num_batches):
                batch_data = data[i: i + batch_size]
                batch_labels = labels[i: i + batch_size]
                batch_data = convertSamplesToEmbeddings(batch_data, index_to_embedding)
                batch_labels = [to_categorical(label, num_classes=len(index_to_embedding)) for label in batch_labels]
                yield np.array(batch_data), np.array(batch_labels)