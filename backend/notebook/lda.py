# %%
# import janome
from gensim.models import LdaModel
from gensim.test.utils import common_corpus


# %%
lda = LdaModel(common_corpus, num_topics=10)

# %%
lda.get_term_topics([0])

# %%


# %%


# %%


# %%


# %%
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

# %%
X, _ = make_multilabel_classification(random_state=0)
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X) 
lda.transform(X[-2:])

# %%
lda.components_

# %%
import numpy
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# %%
news20 = fetch_20newsgroups()

# %%
N = 5000

X_train, X_test, y_train, y_test = train_test_split(
    news20.data[:N], news20.target[:N],
    stratify=news20.target[:N]
)


# %%
X = news20.data[:1000]
y = news20.target[:1000]


# %%
X[0][:100]  # X.shape = (D, V)

# %%
cv = CountVectorizer(min_df=0.04, stop_words="english")


# %%
feature_names = numpy.array(cv.get_feature_names())
feature_names[30:30+10]


# %%
K = 50
beta = 1/len(feature_names)
# beta = 1/K
print("beta:", beta)
lda = LDA(n_components=K, max_iter=50, n_jobs=-1, topic_word_prior=beta)
lda.fit(cv.fit_transform(X, y))


# %%
len(feature_names)


# %%
lda.components_.shape       # (K, )

# %%
for i, component in enumerate(lda.components_[:10]):
    print("component:", i)
    idx = component.argsort()[::-1][:5]
    for j in idx:
        print("\t", feature_names[j], component[j])


# %%


# %%



