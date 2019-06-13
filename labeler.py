# https://medium.com/tensorflow/building-a-text-classification-model-with-tensorflow-hub-and-estimators-3169e7aa568
# Sara Robinson

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

#import tensorflow as tf
#import tensorflow_hub as hub

#from sklearn.preprocessing import MultiLabelBinarizer


#download data


with open('page/25-Hour_Day.pg', 'r') as f:
    mytext = f.read()
    print(mytext)


for f in listdir('page'):
    with open('page/'+f, 'r') as ff:
        mytext = ff.read()

'''
wget.download('https://storage.googleapis.com/movies_data/movies_metadata.csv')
data = pd.read_csv('movies_metadata.csv')

descriptions = data['overview']
genres = data['genres']

top_genres = ['Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'Adventure', 'Science Fiction']

#train/test


train_size = int(len(descriptions) * .8)

train_descriptions = descriptions[:train_size]
train_genres = genres[:train_size]

test_descriptions = descriptions[train_size:]
test_genres = genres[train_size:]


#embeddings

description_embeddings = hub.text_embedding_column(
  "movie_descriptions", 
  module_spec="https://tfhub.dev/google/universal-sentence-encoder/2"
)


#format labels


# Genre lookup, each genre corresponds to an index
top_genres = ['Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'Adventure', 'Science Fiction']

# E.g. Multi-hot label for an action and adventure movie
# [0 0 0 1 0 0 0 1 0]


#label encoder


encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
test_encoded = encoder.transform(test_genres)
num_classes = len(encoder.classes_)


#multi-label head


multi_label_head = tf.contrib.estimator.multi_label_head(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)

#DNN estimater


estimator = tf.contrib.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64,10],
    feature_columns=[description_embeddings]
)

# train input function

# Format our data for the numpy_input_fn
features = {
  "descriptions": np.array(train_descriptions)
}
labels = np.array(train_encoded)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features, 
    labels, 
    shuffle=True, 
    batch_size=32, 
    num_epochs=20
)

# train model

estimator.train(input_fn=train_input_fn)

#eval input function

eval_input_fn = tf.estimator.inputs.numpy_input_fn({"descriptions": np.array(test_descriptions).astype(np.str)}, test_encoded.astype(np.int32), shuffle=False)

estimator.evaluate(input_fn=eval_input_fn)

#test movies

raw_test = [
    "An examination of our dietary choices and the food we put in our bodies. Based on Jonathan Safran Foer's memoir.", # Documentary
    "A teenager tries to survive the last week of her disastrous eighth-grade year before leaving to start high school.", # Comedy
    "Ethan Hunt and his IMF team, along with some familiar allies, race against time after a mission gone wrong." # Action, Adventure
]

#prediction input function

predict_input_fn = tf.estimator.inputs.numpy_input_fn({"descriptions": np.array(raw_test).astype(np.str)}, shuffle=False)

results = estimator.predict(predict_input_fn)

#run predictions

for movie_genres in results:
  top_2 = movie_genres['probabilities'].argsort()[-2:][::-1]
  for genre in top_2:
    text_genre = encoder.classes_[genre]
    print(text_genre + ': ' + str(round(movie_genres['probabilities'][genre] * 100, 2)) + '%')

'''