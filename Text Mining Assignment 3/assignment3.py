import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')
spam_data['target'] = np.where(spam_data['target'] == 'spam', 1, 0)
# print(spam_data.head(10))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def answer_one():
    """
    What percentage of the documents in spam_data are spam?
    Returns a float, the percent value (i.e. ratio * 100).
    """
    # 方法1: 用 target 列（已转为 1=spam, 0=ham）
    spam_count = spam_data['target'].sum()
    total_count = len(spam_data)
    return (spam_count / total_count) * 100


def answer_two():
    """
    Fit X_train with CountVectorizer (default params).
    Return the longest token in the vocabulary (as a string).
    """
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    # 词表：get_feature_names_out() (sklearn >= 1.0) 或 get_feature_names() (旧版)
    try:
        vocab = vectorizer.get_feature_names_out()
    except AttributeError:
        vocab = vectorizer.get_feature_names()
    return max(vocab, key=len)


def answer_three():
    """
    Fit/transform X_train with CountVectorizer (default).
    Fit MultinomialNB with alpha=0.1. Return AUC on transformed test data.
    """
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_vec, y_train)
    # AUC 需要正类( spam=1 )的概率
    y_score = clf.predict_proba(X_test_vec)[:, 1]
    return float(roc_auc_score(y_test, y_score))


def answer_four():
    """
    TfidfVectorizer on X_train; for each feature take max tf-idf over documents.
    Return (series of 20 smallest max tf-idfs, series of 20 largest max tf-idfs).
    Smallest: sorted by tf-idf asc, ties by feature name asc.
    Largest: sorted by tf-idf desc, ties by feature name asc.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train)
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()
    # 每列(每个特征)在所有文档上的最大 tf-idf（稀疏矩阵 axis=0 得到 (1, n_features)）
    max_tfidf = X.max(axis=0).toarray().flatten()
    s = pd.Series(max_tfidf, index=feature_names)
    # 用 (tfidf, feature_name) 排序以满足“同分按名字字典序”
    df = s.reset_index()
    df.columns = ['feature', 'tfidf']
    smallest_df = df.sort_values(by=['tfidf', 'feature'], ascending=[True, True]).head(20)
    largest_df = df.sort_values(by=['tfidf', 'feature'], ascending=[False, True]).head(20)
    smallest_series = smallest_df.set_index('feature')['tfidf']
    largest_series = largest_df.set_index('feature')['tfidf']
    return (smallest_series, largest_series)


def answer_five():
    """
    TfidfVectorizer with min_df=3 (ignore terms with doc freq strictly lower than 3).
    Fit MultinomialNB(alpha=0.1), return AUC on transformed test data.
    """
    vectorizer = TfidfVectorizer(min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_vec, y_train)
    y_score = clf.predict_proba(X_test_vec)[:, 1]
    return float(roc_auc_score(y_test, y_score))


def answer_six():
    """
    Average document length (number of characters) for not spam and spam.
    Returns (average length not spam, average length spam).
    """
    not_spam = spam_data[spam_data['target'] == 0]['text'].str.len().mean()
    spam = spam_data[spam_data['target'] == 1]['text'].str.len().mean()
    return (float(not_spam), float(spam))


def answer_seven():
    """
    TfidfVectorizer(min_df=5) + document length feature, SVC(C=10000).
    probability=False so use decision_function for AUC. Return AUC as float.
    """
    vectorizer = TfidfVectorizer(min_df=5)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    X_train_full = add_feature(X_train_vec, X_train.str.len().values)
    X_test_full = add_feature(X_test_vec, X_test.str.len().values)
    clf = SVC(C=10000)
    clf.fit(X_train_full, y_train)
    y_score = clf.decision_function(X_test_full)
    return float(roc_auc_score(y_test, y_score))


def answer_eight():
    """
    Average number of digits per document for not spam and spam.
    Returns (average # digits not spam, average # digits spam).
    """
    import re
    def count_digits(s):
        return len(re.findall(r'\d', str(s)))
    not_spam = spam_data[spam_data['target'] == 0]['text'].apply(count_digits).mean()
    spam = spam_data[spam_data['target'] == 1]['text'].apply(count_digits).mean()
    return (float(not_spam), float(spam))


def answer_nine():
    """
    TfidfVectorizer(min_df=5, ngram_range=(1,3)) + doc length + digit count.
    LogisticRegression(C=100, max_iter=1000). Return AUC on test data.
    """
    import re
    def count_digits(s):
        return len(re.findall(r'\d', str(s)))
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    train_len = X_train.str.len().values
    train_digits = X_train.apply(count_digits).values
    test_len = X_test.str.len().values
    test_digits = X_test.apply(count_digits).values
    X_train_full = add_feature(add_feature(X_train_vec, train_len), train_digits)
    X_test_full = add_feature(add_feature(X_test_vec, test_len), test_digits)
    clf = LogisticRegression(C=100, max_iter=1000)
    clf.fit(X_train_full, y_train)
    y_score = clf.predict_proba(X_test_full)[:, 1]
    return float(roc_auc_score(y_test, y_score))


def answer_ten():
    """
    Average number of non-word characters (\\W: not letter, digit, underscore) per document.
    Returns (average # non-word chars not spam, average # non-word chars spam).
    """
    import re
    def count_non_word(s):
        return len(re.findall(r'\W', str(s)))
    not_spam = spam_data[spam_data['target'] == 0]['text'].apply(count_non_word).mean()
    spam = spam_data[spam_data['target'] == 1]['text'].apply(count_non_word).mean()
    return (float(not_spam), float(spam))


def answer_eleven():
    """
    First 2000 rows: CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=(2,5))
    + length_of_doc, digit_count, non_word_char_count.
    LogisticRegression(C=100, max_iter=1000). Return (AUC, 10 smallest coefs, 10 largest coefs).
    """
    import re
    def count_digits(s):
        return len(re.findall(r'\d', str(s)))
    def count_non_word(s):
        return len(re.findall(r'\W', str(s)))
    X_train_2000 = X_train.iloc[:2000]
    y_train_2000 = y_train.iloc[:2000]
    vectorizer = CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=(2, 5))
    X_train_vec = vectorizer.fit_transform(X_train_2000)
    X_test_vec = vectorizer.transform(X_test)
    train_len = X_train_2000.str.len().values
    train_digits = X_train_2000.apply(count_digits).values
    train_non_word = X_train_2000.apply(count_non_word).values
    test_len = X_test.str.len().values
    test_digits = X_test.apply(count_digits).values
    test_non_word = X_test.apply(count_non_word).values
    X_train_full = add_feature(add_feature(add_feature(X_train_vec, train_len), train_digits), train_non_word)
    X_test_full = add_feature(add_feature(add_feature(X_test_vec, test_len), test_digits), test_non_word)
    clf = LogisticRegression(C=100, max_iter=1000)
    clf.fit(X_train_full, y_train_2000)
    y_score = clf.predict_proba(X_test_full)[:, 1]
    auc = float(roc_auc_score(y_test, y_score))
    coef = clf.coef_[0]
    smallest_coefs = sorted(coef)[:10]
    largest_coefs = sorted(coef, reverse=True)[:10]
    return (auc, smallest_coefs, largest_coefs)


# 测试
if __name__ == '__main__':
    result = answer_one()
    print(f'Question 1 - 垃圾短信占比: {result:.2f}%')
    print(f'answer_one() 返回值: {result}')
    print()
    result2 = answer_two()
    print(f'Question 2 - 最长 token: {result2!r} (长度 {len(result2)})')
    print(f'answer_two() 返回值: {result2!r}')
    print()
    result3 = answer_three()
    print(f'Question 3 - AUC: {result3}')
    print(f'answer_three() 返回值: {result3}')
    print()
    small4, large4 = answer_four()
    print('Question 4 - 20 smallest max tf-idf:')
    print(small4)
    print()
    print('Question 4 - 20 largest max tf-idf:')
    print(large4)
    print()
    result5 = answer_five()
    print(f'Question 5 - AUC: {result5}')
    print(f'answer_five() 返回值: {result5}')
    print()
    result6 = answer_six()
    print(f'Question 6 - (非垃圾平均字符数, 垃圾平均字符数): {result6}')
    print(f'answer_six() 返回值: {result6}')
    print()
    result7 = answer_seven()
    print(f'Question 7 - AUC: {result7}')
    print(f'answer_seven() 返回值: {result7}')
    print()
    result8 = answer_eight()
    print(f'Question 8 - (非垃圾平均数字个数, 垃圾平均数字个数): {result8}')
    print(f'answer_eight() 返回值: {result8}')
    print()
    result9 = answer_nine()
    print(f'Question 9 - AUC: {result9}')
    print(f'answer_nine() 返回值: {result9}')
    print()
    result10 = answer_ten()
    print(f'Question 10 - (非垃圾平均非单词字符数, 垃圾平均非单词字符数): {result10}')
    print(f'answer_ten() 返回值: {result10}')
    print()
    auc11, small11, large11 = answer_eleven()
    print(f'Question 11 - AUC: {auc11}')
    print(f'10 smallest coefs: {small11}')
    print(f'10 largest coefs: {large11}')
    print(f'answer_eleven() 返回值: (AUC, 10 smallest, 10 largest) = ({auc11}, ...)')
