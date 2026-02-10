import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import string

nltk.download("punkt")
nltk.download("stopwords")

corpus = [
    "King is a strong man",
    "Queen is a wise woman",
    "Boy is a young man",
    "Girl is a young woman",
    "King and Queen are royal",
    "Man and woman are human"
]

stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)

processed_sentences = []

for sentence in corpus:
    tokens = word_tokenize(sentence.lower())
    tokens = [w for w in tokens if w not in stop_words and w not in punct]
    processed_sentences.append(tokens)

cbow_model = Word2Vec(
    sentences=processed_sentences,
    vector_size=50,
    window=2,
    min_count=1,
    sg=0,
    epochs=500
)

skipgram_model = Word2Vec(
    sentences=processed_sentences,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1,
    epochs=500
)

print("CBOW similarity (king, queen):",
      cbow_model.wv.similarity("king", "queen"))

print("Skip-gram similarity (king, queen):",
      skipgram_model.wv.similarity("king", "queen"))

print("CBOW similarity (man, woman):",
      cbow_model.wv.similarity("man", "woman"))

print("Skip-gram similarity (man, woman):",
      skipgram_model.wv.similarity("man", "woman"))

print("\nCBOW most similar to king:")
print(cbow_model.wv.most_similar("king"))

print("\nSkip-gram most similar to king:")
print(skipgram_model.wv.most_similar("king"))

print("\nCBOW analogy:")
print(cbow_model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
))

print("\nSkip-gram analogy:")
print(skipgram_model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
))
