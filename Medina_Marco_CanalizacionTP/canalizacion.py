# Realizar una “canalización” o “pipeline” para analizar el siguiente corpus CorpusLenguajes.txt
# -Aplicar stop_word,
# -Lematización
# -Tf-Idf
# -Mostrar el corpus preparado
# -Mostrar la matriz TF-IDF generada
# -Mostrar el vocabulario generado
# Analizar el mismo y redactar un informe con las conclusiones obtenidas.
# -Obtener las jerarquía de 6 palabras mas usadas en todo el corpus
# -La palabra menos utilizada
# -Las palabras mas repetidas en la misma oración
# -Imprimir el gráfico de Distribución de Frecuencia.

import nltk
nltk.download("stopwords")
nltk.download('wordnet')
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import FreqDist



lemmatizer=WordNetLemmatizer()
vectorizer= TfidfVectorizer()

def get_wordnet_pos(word):
    """MapPOStagtofirstcharacterlemmatize()accepts"""
    tag=nltk.pos_tag([word])[0][1][0].upper()
    tag_dict={"J":wordnet.ADJ,
    "N":wordnet.NOUN,
    "V":wordnet.VERB,
    "R":wordnet.ADV}
    return tag_dict.get(tag,wordnet.NOUN)

def quitarStopwords_eng(texto):
    ingles=stopwords.words("english")
    texto_limpio=[w.lower()for w in texto if w.lower()not in ingles
    and w not in string.punctuation
    and w not in["'s",'|','--',"''","``"]]
    return texto_limpio

def lematizar(texto):
    texto_lema=[lemmatizer.lemmatize(w, get_wordnet_pos(w))for w in texto]
    return texto_lema


corpusTP=PlaintextCorpusReader(".","CorpusLenguajes.txt")


texto=corpusTP.raw()
texto_tokenizado= word_tokenize(texto)
texto_lematizado= (lematizar(quitarStopwords_eng(texto_tokenizado)))
corpus2TP=" ".join(texto_lematizado)

print(texto)
print(" ")
print("TEXTO SIN STOP_WORD Y LEMATIZADO: ")
print(texto_lematizado)
print(" ")

texto_vectorizado=vectorizer.fit_transform([corpus2TP])
frecuencia = FreqDist(corpusTP.words())

print("Se aplica TF-IDF:")
print(texto_vectorizado)
print(" ")
print("Matriz TF-IDF:")
print(texto_vectorizado.toarray())
print("\nVocabulario:")
print(vectorizer.get_feature_names_out())


for mc in frecuencia.most_common(6):
    print(mc)

frecuencia.plot(6,show=True)
