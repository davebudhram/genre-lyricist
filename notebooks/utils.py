import nltk

SENTENCE_BEGIN = '<s>'
SENTENCE_END = '</s>'
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