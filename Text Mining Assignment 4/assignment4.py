import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    tokens = nltk.word_tokenize(doc)
    pos_tags = nltk.pos_tag(tokens)

    synsets = []
    for token, tag in pos_tags:
        wordnet_tag = convert_tag(tag)
        if wordnet_tag is None:
            syns = wn.synsets(token)
        else:
            syns = wn.synsets(token, wordnet_tag)
        if syns:
            synsets.append(syns[0])

    return synsets


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.7333333333333333
    """
    largest_scores = []

    for syn1 in s1:
        scores = []
        for syn2 in s2:
            sim = syn1.path_similarity(syn2)
            if sim is not None:
                scores.append(sim)
        if scores:
            largest_scores.append(max(scores))

    if not largest_scores:
        return 0.0

    return sum(largest_scores) / len(largest_scores)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


# ---- 验证测试 ----
if __name__ == '__main__':
    # 测试 doc_to_synsets
    result = doc_to_synsets('Fish are friends.')
    print("doc_to_synsets('Fish are friends.'):", result)
    # 期望: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]

    # 测试 similarity_score
    synsets1 = doc_to_synsets('I like cats')
    synsets2 = doc_to_synsets('I like dogs')
    score = similarity_score(synsets1, synsets2)
    print("similarity_score('I like cats', 'I like dogs'):", score)
    # 期望: 0.7333333333333333

    # 测试 document_path_similarity
    sim = document_path_similarity('I like cats', 'I like dogs')
    print("document_path_similarity('I like cats', 'I like dogs'):", sim)

    # 在 paraphrases 数据集上运行
    paraphrases = pd.read_csv(r'C:\Users\23059\Downloads\Text Mining Assignment 4\paraphrases.csv')
    print("\nparaphrases 数据集前几行:")
    print(paraphrases.head())

    # 计算每对文档的相似度
    paraphrases['similarity'] = paraphrases.apply(
        lambda row: document_path_similarity(row['D1'], row['D2']), axis=1
    )
    print("\n加入相似度分数后的数据集前几行:")
    print(paraphrases.head(10))
