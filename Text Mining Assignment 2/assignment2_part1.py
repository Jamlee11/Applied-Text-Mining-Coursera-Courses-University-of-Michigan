# Assignment 2 - Introduction to NLTK
# Part 1 - Analyzing Plots Summary Text

import nltk
import pandas as pd
import numpy as np

# 首次运行需下载 NLTK 数据（若已下载可注释掉）
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('words', quiet=True)

# 使用你的数据路径
DATA_PATH = r"C:\Users\23059\Downloads\Text Mining Assignment 2\plots.txt"

# 如需使用 nltk 数据可取消下一行注释并指定路径
# nltk.data.path.append("assets/")

with open(DATA_PATH, 'rt', encoding="utf8") as f:
    plots_raw = f.read()

plots_tokens = nltk.word_tokenize(plots_raw)
text1 = nltk.Text(plots_tokens)


def example_one():
    """How many tokens (words and punctuation symbols) are in text1?"""
    return len(nltk.word_tokenize(plots_raw))  # or alternatively len(text1)


def example_two():
    """How many unique tokens (unique words and punctuation) does text1 have?"""
    return len(set(nltk.word_tokenize(plots_raw)))  # or alternatively len(set(text1))


def example_three():
    """After lemmatizing the verbs, how many unique tokens does text1 have?"""
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w, 'v') for w in text1]
    return len(set(lemmatized))


def answer_one():
    """
    What is the lexical diversity of the given text input?
    (i.e. ratio of unique tokens to the total number of tokens)
    """
    total_tokens = len(text1)  # 或 len(plots_tokens)
    unique_tokens = len(set(text1))  # 或 len(set(plots_tokens))
    lexical_diversity = unique_tokens / total_tokens
    return float(lexical_diversity)


def answer_two():
    """
    What percentage of tokens is 'love' or 'Love'?
    """
    total = len(text1)
    count = sum(1 for w in text1 if w in ('love', 'Love'))
    return float(count / total * 100)


def answer_three():
    """
    What are the 20 most frequently occurring (unique) tokens in the text?
    Return a list of 20 tuples (token, frequency), sorted in descending order of frequency.
    """
    fdist = nltk.FreqDist(text1)
    return fdist.most_common(20)


def answer_four():
    """
    What tokens have a length of greater than 5 and frequency of more than 200?
    Return an alphabetically sorted list of the matching tokens.
    """
    fdist = nltk.FreqDist(text1)
    matching = [token for token in fdist if len(token) > 5 and fdist[token] > 200]
    return sorted(matching)


def answer_five():
    """
    Find the longest token in text1 and that token's length.
    Return a tuple (longest_word, length).
    """
    longest = max(text1, key=len)
    return (longest, len(longest))


def answer_six():
    """
    What unique words (isalpha()) have a frequency of more than 2000?
    Return a list of tuples (frequency, word) sorted in descending order of frequency.
    """
    fdist = nltk.FreqDist(text1)
    words_over_2000 = [(fdist[w], w) for w in fdist if w.isalpha() and fdist[w] > 2000]
    return sorted(words_over_2000, key=lambda x: x[0], reverse=True)


def answer_seven():
    """
    Join text1 tokens with whitespace, sentence-tokenize with nltk.sent_tokenize,
    then report the average number of whitespace-separated tokens per sentence.
    """
    joined = " ".join(text1)
    sentences = nltk.sent_tokenize(joined)
    total_tokens = sum(len(s.split()) for s in sentences)
    num_sentences = len(sentences)
    return float(total_tokens / num_sentences)


def answer_eight():
    """
    What are the 5 most frequent parts of speech in text1?
    Return a list of tuples (part_of_speech, frequency) sorted in descending order of frequency.
    """
    pos_tagged = nltk.pos_tag(text1)
    pos_tags = [tag for (word, tag) in pos_tagged]
    fdist = nltk.FreqDist(pos_tags)
    return fdist.most_common(5)


# ========== Part 2 - Spelling Recommender ==========
from nltk.corpus import words

correct_spellings = words.words()


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    """
    Recommender using Jaccard distance on trigrams.
    For each misspelled word, find the word in correct_spellings that starts with
    the same letter and has the shortest Jaccard distance (on trigrams).
    """
    result = []
    for misspelled in entries:
        first_char = misspelled[0].lower()
        candidates = [w for w in correct_spellings if w.lower().startswith(first_char)]
        trigrams_m = set(nltk.ngrams(misspelled.lower(), 3))
        best_word = min(
            candidates,
            key=lambda w: nltk.jaccard_distance(trigrams_m, set(nltk.ngrams(w.lower(), 3)))
        )
        result.append(best_word)
    return result


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    """
    Recommender using Jaccard distance on 4-grams.
    Same as answer_nine but with n=4 for ngrams.
    """
    result = []
    for misspelled in entries:
        first_char = misspelled[0].lower()
        candidates = [w for w in correct_spellings if w.lower().startswith(first_char)]
        ngrams_m = set(nltk.ngrams(misspelled.lower(), 4))
        best_word = min(
            candidates,
            key=lambda w: nltk.jaccard_distance(ngrams_m, set(nltk.ngrams(w.lower(), 4)))
        )
        result.append(best_word)
    return result


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    """
    Recommender using edit distance with transpositions (Damerau–Levenshtein).
    For each misspelled word, find the word in correct_spellings that starts with
    the same letter and has the smallest edit distance (transpositions=True).
    """
    result = []
    for misspelled in entries:
        first_char = misspelled[0].lower()
        candidates = [w for w in correct_spellings if w.lower().startswith(first_char)]
        best_word = min(
            candidates,
            key=lambda w: nltk.edit_distance(misspelled.lower(), w.lower(), transpositions=True)
        )
        result.append(best_word)
    return result


if __name__ == "__main__":
    print("Example 1 - Total tokens:", example_one())
    print("Example 2 - Unique tokens:", example_two())
    print("Example 3 - Unique tokens after lemmatizing verbs:", example_three())
    print("Question 1 - Lexical diversity:", answer_one())
    print("Question 2 - Percentage of 'love' or 'Love':", answer_two())
    print("Question 3 - 20 most frequent tokens:", answer_three())
    print("Question 4 - tokens len>5 & freq>200 (alphabetically):", answer_four())
    print("Question 5 - longest token:", answer_five())
    print("Question 6 - words (isalpha) with freq>2000:", answer_six())
    print("Question 7 - avg tokens per sentence:", answer_seven())
    print("Question 8 - 5 most frequent POS:", answer_eight())
    print("Question 9 - spelling (Jaccard trigram):", answer_nine())
    print("Question 10 - spelling (Jaccard 4-gram):", answer_ten())
    print("Question 11 - spelling (edit dist + transpositions):", answer_eleven())
