"""
Word2Vec 사용
gensim : word2vec을 구현한 pypthon module
"""

import pandas as pd

train =  pd.read_csv('DATA/labeledTrainData.tsv', header = 0, delimiter='\t', quoting =3)
test =  pd.read_csv('DATA/testData.tsv', header = 0, delimiter='\t', quoting =3)
unlabeled_train = pd.read_csv('DATA/unlabeledTrainData.tsv', header = 0, delimiter='\t', quoting=3)

from KaggleWord2VecUtility import KaggleWord2VecUtility


sentences = []
for review in train['reivew'] :
    sentences += KaggleWord2VecUtility.review_to_sentences(review, remove_stopwords=False)

for review in unlabeled_train['review']:
    sentences += KaggleWord2VecUtility.review_to_sentences(review, remove_stopwords=False)


import logging
logging.basicConfig(
    format = '%(asctime)s : %(levelname)s : %(message)s',
    level = logging.INFO
)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3


from gensim.models import word2vec

# word2vec 모델 만들어서 학습.
model = word2vec.WordVec(sentences,
                         workers = num_workers,
                         size = num_features,
                         min_count = min_word_count,
                         window = context,
                         sample = downsampling)


# 학습 완료 후 필요없는 memory unload.
model.init_sims(replace=True)

model_name = '300features_40minwords_10text'
model.save(model_name)


# 학습된 모델 결과 탐색

# 유사도 없는 단어 추출
print(model.wv.dosent_match('man woman child kitchen'.spilt())) # kitchen만 사람이 아님.
print(model.wv.dosent_match('france england germany berlin'.split())) # belin만 나라이름이 아님.

# 가장 유사한 단어 추출
print(model.wv.most_similar('man'))

print(model.wv.most_similar('queen'))



# Word2Vec 벡터화한 단어를 t-SNE로 시각화하기

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib





