import pandas as pd

"""
 https://github.com/corazzon/KaggleStruggle/blob/master/word2vec-nlp-tutorial/KaggleWord2VecUtility.py
header = 0 은 FILE의 첫 번째 줄에 열 이름이 있음을 나타낸다.
delimiter = \t 은 필드가 tab으로 구분되어 있다는 것을 의미한다.
quoting = 3 은 쌍따옴표를 무시하도록 한다.

"""

# quouting 관련 flag들은 다음과 같다.

# QUOTE_MINIMAL (0)
# QUOTE_ALL (1)
# QUOTE_NONNUMERIC (2)
# QUOTE_NONE (3)

# 레이블인 sentiment가 있는 학습 데이터
train = pd.read_csv('DATA/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

# 레이블이 없는 test data
test = pd.read_csv('DATA/testData.tsv', header=0, delimiter="\t", quoting=3)

# 전체 tuple의 갯수와 열의 갯수를 반환한다.
train.shape

# train.tail(n) 은 n개의 마지막 데이터들을 가져온다.

# 열의 이름들을 반환한다.
train.columns.values

# train에 불러온 파일의 정보를 표시한다.
train.info()

# 숫자 값으로 이루어진 열에 대한 평균 / 중간 / 최소값 등을 표시한다.
train.describe()

# sentiment 열의 값이 어떻게 이루어져있고 각각의 값이 총 몇 번 나타나는지 반환한다.
train['sentiment'].value_counts()

# review 열의 첫 번째 data를 가져옴.
train['review'][0]

"""
여기까지가 Pandas의 기본 사용법 __ Pandas는 csv, tsv 파일등을 불러올 때 사용한다고 이해함.
More to Learn
"""

# HTML 태그 처리를 위해 BeautifulSoup import.

from bs4 import BeautifulSoup

example1 = BeautifulSoup(train['review'][0], "html5lib")
print(train['review'][0][:700])

# get_text()는 html 태그들을 제거하여 text만 반환.
# example1.get_text()


# 정규표현식 이용해 특수문자 제거 (Noise 처리)
import re

letters_only = re.sub('[^a-zA-Z]', ' ', example1.get_text())

# 모든 문자를 소문자로 변환 ( 대문자와 소문자로 구성된 단어를 다르게 인식하므로 )
lower_case = letters_only.lower()
# 단어를 전부 split한다.
words = lower_case.split()

# 영어 불용어 제거를 위해 nltk 이용 (아쉽게도 한국어는 없다.)
import nltk
from nltk.corpus import stopwords

# 불용어를 체거하여 다시 words list를 정리함. (조건부가 true일 경우에만 list에 추가한다!! ""list comprehension"")
words = [w for w in words if not stopwords.words('english')]

# Stemming 해줌 (관련 내용은 iPAD 참고)
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
words = [stemmer.stem(w) for w in words]

"""
Lemmatization으로 처리
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatize()

words = [wordnet_lemmatizer.lemmatize(w) for w in words]
"""


# 위의 과정들을 하나의 function으로 처리
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    # 불용어들을 set으로 변환해버림.
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return (' '.join(stemming_words))


clean_review = review_to_words(train['review'][0])
print(clean_review)

""" 
모든 데이터를 처리해야함...
"""

# review의 개수
num_reivews = train['review'].size

# 이걸 하나의 processor만 이용해서 하면 매우 오래걸린다.... -> multi-processing을 한다!!!

from multiprocessing import Pool
import numpy as np


# 참고 : https://gist.github.com/yong27/7869662
# http://www.racketracer.com/2016/07/06/pandas-in-parallel/
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


# worker를 4개로 지정헀으므로 4개의 thread가 돌 것이다.
clean_train_reviews = apply_by_multiprocessing(train['reviews'], review_to_words, workers=4)
clean_test_reviews = apply_by_multiprocessing(test['reviews'], review_to_words, workers=4)



# BOW model을 이용하기위해서 각각의 word가 몇 번 나타났는지 확인해야할 필요가 있다.
# 이를 위해 CountVectorizer를 이용한다.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

vectorizer = CountVectorizer( analyzer='word',
                              tokenizer=None,
                              preprocessor=None,
                              stop_words= None,
                              min_df = 2, # 토큰이 나타날 최소 문서 개수
                              ngram_range=(1,3),
                              max_features = 20000
                            )

# piipeline을 이용해서 속도를 단축함.
pipleline = Pipeline([('vect', vectorizer),])
train_data_features = pipleline.fit_transform(clean_train_reviews)

print(train_data_features.shape)

vocab = vectorizer.get_feature_names()

"""
위의 CountVctorizer 파트를 이해하지 못함... 
"""

# vector화된 feature를 확인
import numpy as np
dist = np.sum(train_data_features, axis= 0)

for tag, count in zip(vocab, dist):
    print(count, tag)

pd.DataFrame(dist, columns=vocab)



from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, n_jobs= -1, random_state=2018)
# n_estimators가 클 수록 정확도가 올라간다.
# n_jobs는 몇 개의 코어를 동시에 돌릴 것인지... (-1이면 모든 CPU 코어 사용. 2,3,.... )
# random_state는 난수이다. random forest를 돌릴 때마다 다른 형태를 이용하도록... 상수를 지정해서 항상 같은 결과를 갖게 함으로써 param tuning시 어떻게 달라지는지 알 수 있다.


forest = forest.fit(train_data_features, train['sentiment'])

from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(forest, train_data_features, train['sentiment'], cv=10, scoring='roc_auc')) # roc 기법 사용해서 scoring.


# 이제 test data 이용해서 scoring
test_data_features = pipleline.fit_transform(clean_test_reviews)

result = forest.predict(test_data_features) # 감성분석 결과가 result에 저장된다.

output = pd.DataFrame(data={'id':test['id'], 'sentiment':result})

output.to_csv('DATA/tutorial_1_BOW_{0:.5f}.csv'.format(score), index=False, quoting=3)




