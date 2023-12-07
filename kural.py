import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import icu
import string
df = pd.read_csv("Thirukural.csv")

def prep():
    documents = df['Explanation'].values.tolist()
    lemmatizer = WordNetLemmatizer()

    nltk_stopwords = set(stopwords.words('english'))
    nltk_stopwords.update(set(string.punctuation))
    additional_stopwords = ['and', 'but', 'or', 'because', 'as', 'if', 'when']
    nltk_stopwords.update(set(additional_stopwords))
    nltk_stopwords.add('explanation')

    texts = [
        ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(document.lower()) if word not in nltk_stopwords])
        for document in documents
    ]
    return texts


def findKural(user_input, texts):
    lemmatizer = WordNetLemmatizer()
    # user_input = "FRIENDS"
    user_input = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(user_input.lower())])
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    user_input_vector = tfidf_vectorizer.transform([user_input])

    cosine_similarities = linear_kernel(user_input_vector, tfidf_matrix)

    similarity_threshold = 0.3

    kural_indices = np.where(cosine_similarities > similarity_threshold)[1]

    sorted_kurals = sorted(zip(kural_indices, cosine_similarities[0, kural_indices]), key=lambda x: x[1], reverse=True)

    transliterator = icu.Transliterator.createInstance("Tamil-Latin")

    top_5_kurals = []
    for kural_index, similarity in sorted_kurals[:5]:
        section_name = df.iloc[kural_index]['Section Name']
        chapter_name = df.iloc[kural_index]['Chapter Name']
        explanation = df.iloc[kural_index]['Explanation']
        verse = df.iloc[kural_index]['Verse']

        # Transliterate the section and chapter names from Tamil to English
        section_name_transliterated = transliterator.transliterate(section_name)
        chapter_name_transliterated = transliterator.transliterate(chapter_name)
        VerseTrans = transliterator.transliterate(verse)

        top_5_kurals.append({
            'Verse': verse,
            'Verse Transliterated': VerseTrans,
            'Section Name (Tamil)': section_name,
            'Chapter Name (Tamil)': chapter_name,
            'Explanation (Tamil)': explanation,
            'Section Name (Transliterated)': section_name_transliterated,
            'Chapter Name (Transliterated)': chapter_name_transliterated,
            'Explanation (English)': explanation,
            'Similarity': similarity
        })
    return top_5_kurals


def show_kural(top_5_kurals):
    for i, kural in enumerate(top_5_kurals):
        print(f"Kural {i + 1}")
        print(f"Section Name (Tamil): {kural['Section Name (Tamil)']}")
        print(f"Section Name (Transliterated): {kural['Section Name (Transliterated)']}")
        print(f"Chapter Name (Tamil): {kural['Chapter Name (Tamil)']}")
        print(f"Chapter Name (Transliterated): {kural['Chapter Name (Transliterated)']}")
        print(
            "_________________________________________________________________________________________________________________")
        print(f"Verse (Tamil): {kural['Verse']}")
        print(f"Verse (Tanglish): {kural['Verse Transliterated']}")
        print(
            "_________________________________________________________________________________________________________________")
        print(f"Explanation (English): {kural['Explanation (English)']}")
        print(f"Similarity: {kural['Similarity']}")
        print("")


# texts = prep()
# kurals = findKural("FRIENDS", texts)
# show_kural(kurals)

# df = pd.read_csv('Thirukural.csv')
# documents = df['Explanation'].values.tolist()
# lemmatizer = WordNetLemmatizer()
#
# nltk_stopwords = set(stopwords.words('english'))
# nltk_stopwords.update(set(string.punctuation))
# additional_stopwords = ['and', 'but', 'or', 'because', 'as', 'if', 'when']
# nltk_stopwords.update(set(additional_stopwords))
# nltk_stopwords.add('explanation')
#
# texts = [
#     ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(document.lower()) if word not in nltk_stopwords])
#     for document in documents
# ]
#
# ##############
# user_input = "FRIENDS"
# user_input = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(user_input.lower())])
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
# user_input_vector = tfidf_vectorizer.transform([user_input])
#
# cosine_similarities = linear_kernel(user_input_vector, tfidf_matrix)
# # for index, similarity in enumerate(cosine_similarities[0]):
# #     print(f"Similarity with Document {index + 1}: {similarity}")
#
# similarity_threshold = 0.3
#
# kural_indices = np.where(cosine_similarities > similarity_threshold)[1]
# print(kural_indices)
#
# sorted_kurals = sorted(zip(kural_indices, cosine_similarities[0, kural_indices]), key=lambda x: x[1], reverse=True)
#
#
# transliterator = icu.Transliterator.createInstance("Tamil-Latin")
#
#
# top_5_kurals = []
# for kural_index, similarity in sorted_kurals[:5]:
#     section_name = df.iloc[kural_index]['Section Name']
#     chapter_name = df.iloc[kural_index]['Chapter Name']
#     explanation = df.iloc[kural_index]['Explanation']
#     verse= df.iloc[kural_index]['Verse']
#
#     # Transliterate the section and chapter names from Tamil to English
#     section_name_transliterated = transliterator.transliterate(section_name)
#     chapter_name_transliterated = transliterator.transliterate(chapter_name)
#     VerseTrans=transliterator.transliterate(verse)
#
#     top_5_kurals.append({
#         'Verse':verse,
#         'Verse Transliterated': VerseTrans,
#         'Section Name (Tamil)': section_name,
#         'Chapter Name (Tamil)': chapter_name,
#         'Explanation (Tamil)': explanation,
#         'Section Name (Transliterated)': section_name_transliterated,
#         'Chapter Name (Transliterated)': chapter_name_transliterated,
#         'Explanation (English)': explanation,
#         'Similarity': similarity
#     })
#
# for i, kural in enumerate(top_5_kurals):
#     print(f"Kural {i + 1}")
#     print(f"Section Name (Tamil): {kural['Section Name (Tamil)']}")
#     print(f"Section Name (Transliterated): {kural['Section Name (Transliterated)']}")
#     print(f"Chapter Name (Tamil): {kural['Chapter Name (Tamil)']}")
#     print(f"Chapter Name (Transliterated): {kural['Chapter Name (Transliterated)']}")
#     print("_________________________________________________________________________________________________________________")
#     print(f"Verse (Tamil): {kural['Verse']}")
#     print(f"Verse (Tanglish): {kural['Verse Transliterated']}")
#     print("_________________________________________________________________________________________________________________")
#     print(f"Explanation (English): {kural['Explanation (English)']}")
#     print(f"Similarity: {kural['Similarity']}")
#     print("")
