#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from spacy.lang.en import English
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate
from keras.layers import Layer, InputSpec
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import pickle
import json
import innvestigate
import pprint
import os
import math
import gensim.downloader as api
from re import sub
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity

dataset_name = f"./all_data_YelpSmall500.pickle"
output_filename = f"Yelp CNN.json"

# In[4]:


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

data = pickle.load(open(dataset_name, "rb"))


# In[5]:


print(type(data))


# In[6]:


print(data.keys())


# In[7]:


print(data["class_names"])


# In[5]:


class_names = data["class_names"]
y_test = data["y_test"]


# In[6]:


nlp = English()
tokenizer = nlp.tokenizer


# In[7]:


f = open("./glove.6B.300d.txt",'r', encoding = 'utf-8')
glove = {}
for line in tqdm(f):
    splitLine = line.split()
    glove[splitLine[0]] = np.array([float(val) for val in splitLine[1:]])


# In[8]:


vocab = ['PAD','UNK'] + list(glove.keys())
word2index = dict([(w, idx) for idx, w in enumerate(vocab)])
index2word = dict([(idx, w) for idx, w in enumerate(vocab)])

# Initialise vectors for PAD and UNK
pad_vector = np.zeros(300)
mean_vector = np.mean(np.array(list(glove.values())), axis = 0)

# Create the embedding matrix
embedding_matrix = np.concatenate(([pad_vector, mean_vector], np.array([glove[index2word[idx]] for idx in range(2, len(index2word))])), axis = 0)
vocab_size = len(vocab)

X_train = []
for text in tqdm(data["text_train"]):
    tokens = tokenizer(text)
    vector = []
    for token in tokens:
        if token.text.lower() in vocab:
            vector.append(word2index[token.text.lower()])
        elif token.text.lower().strip() != '':
            vector.append(1) # UNK
        if len(vector) >= 150:
            break
    X_train.append(vector)
X_train = np.array(X_train)
X_train = pad_sequences(X_train, 150, dtype="int32", padding="post", truncating="post", value=0.0)

X_validate = []
for text in tqdm(data["text_validate"]):
    tokens = tokenizer(text)
    vector = []
    for token in tokens:
        if token.text.lower() in vocab:
            vector.append(word2index[token.text.lower()])
        elif token.text.lower().strip() != '':
            vector.append(1) # UNK
        if len(vector) >= 150:
            break
    X_validate.append(vector)
X_validate = np.array(X_validate)
X_validate = pad_sequences(X_validate, 150, dtype="int32", padding="post", truncating="post", value=0.0)

X_test = []
for text in tqdm(data["text_test"]):
    tokens = tokenizer(text)
    vector = []
    for token in tokens:
        if token.text.lower() in vocab:
            vector.append(word2index[token.text.lower()])
        elif token.text.lower().strip() != '':
            vector.append(1) # UNK
        if len(vector) >= 150:
            break
    X_test.append(vector)
X_test = np.array(X_test)
X_test = pad_sequences(X_test, 150, dtype="int32", padding="post", truncating="post", value=0.0)


# In[9]:


class MaskedDense(Layer):

    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.glorot_uniform()
        self.bias_initializer = keras.initializers.Zeros()
        self.mask_initializer = keras.initializers.Ones()
        super(MaskedDense, self).__init__(**kwargs)

    def get_config(self):
        config = super(MaskedDense, self).get_config().copy()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "mask_initializer": self.mask_initializer
        })
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel')

        # The mask is not trainable
        self.mask = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.mask_initializer,
                                      trainable=False,
                                      name='mask')

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        super(MaskedDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.mask)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def set_mask(self, value, feature_idx, class_idx = None):
        weights = K.get_value(self.mask)
        assert feature_idx >= 0 and feature_idx < weights.shape[0], f"Feature index out of bound [0, ..., {weights.shape[0]-1}] -- {feature_idx} given"
        if class_idx is not None:
            if isinstance(class_idx, list):
                for idx in class_idx:
                    assert idx >= 0 and idx < weights.shape[1], f"Class index out of bound [0, ..., {weights.shape[1]-1}] -- {idx} given"
                    weights[feature_idx,idx] = value
            elif isinstance(class_idx, int):
                idx = class_idx
                assert idx >= 0 and idx < weights.shape[1], f"Class index out of bound [0, ..., {weights.shape[1]-1}] -- {idx} given"
                weights[feature_idx,idx] = value
        else:
            weights[feature_idx,:] = value
        K.set_value(self.mask, weights)

    def disable_mask(self, feature_idx, class_idx = None):
        self.set_mask(value = 0, feature_idx = feature_idx, class_idx = class_idx)

    def enable_mask(self, feature_idx, class_idx = None):
        self.set_mask(value = 1, feature_idx = feature_idx, class_idx = class_idx)

    def get_masked_weights(self):
        return K.get_value(self.mask) * K.get_value(self.kernel)


# In[10]:


text_input = Input(shape=(None,), dtype="int32")
embedded_text = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=150, trainable=True)(text_input)

filters = [(10, 2), (10, 3), (10, 4)]
filter_layers = [Conv1D(f[0], f[1], activation='relu', trainable=True)(embedded_text) for f in filters]
max_pool_layers = [GlobalMaxPool1D()(result) for result in filter_layers]

concatenated = concatenate(max_pool_layers,axis=-1)

ans = MaskedDense(len(class_names), activation='softmax')(concatenated)

model = Model(text_input, ans)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

y_train_onehot, y_validate_onehot = to_categorical(data['y_train']), to_categorical(data['y_validate'])

checkpointer = ModelCheckpoint(filepath="trained_CNN.h5", verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

# Train the model
model.fit(X_train, y_train_onehot, verbose = 2, epochs=300, batch_size=128, callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))


# In[11]:


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

final_results = {}

prediction_test_onehot = model.predict(X_test, batch_size=128)
prediction_test = prediction_test_onehot.argmax(axis=1).squeeze()
results = {'per_class': {}, 'total': {}}
y_true, y_predict = np.array(y_test), np.array(prediction_test)
for idx, cname in enumerate(class_names):
    TP = np.sum(np.logical_and((y_true == y_predict), (y_true == idx)))
    class_precision = TP / np.sum(y_predict == idx)
    class_recall = TP / np.sum(y_true == idx)
    class_f1 = 2*class_precision*class_recall / (class_precision + class_recall)
    results['per_class'][idx] = {'class_name': cname,
                                 'true_positive': TP,
                                 'all_positive': np.sum(y_predict == idx),
                                 'all_true': np.sum(y_true == idx),
                                 'class_precision': class_precision,
                                 'class_recall': class_recall,
                                 'class_f1': class_f1}

results['total']['macro_precision'] = np.mean([results['per_class'][idx]['class_precision'] for idx in results['per_class']])
results['total']['macro_recall'] = np.mean([results['per_class'][idx]['class_recall'] for idx in results['per_class']])
results['total']['macro_f1'] = 2 * results['total']['macro_precision'] * results['total']['macro_recall'] / (results['total']['macro_precision'] + results['total']['macro_recall'])
results['total']['accuracy'] = sum([results['per_class'][idx]['true_positive'] for idx in results['per_class']]) / len(y_true)
results['total']['micro_precision'] = np.sum([results['per_class'][idx]['true_positive'] for idx in results['per_class']]) / np.sum([results['per_class'][idx]['all_positive'] for idx in results['per_class']])
results['total']['micro_recall'] = np.sum([results['per_class'][idx]['true_positive'] for idx in results['per_class']]) / np.sum([results['per_class'][idx]['all_true'] for idx in results['per_class']])
results['total']['micro_f1'] = 2 * results['total']['micro_precision'] * results['total']['micro_recall'] / (results['total']['micro_precision'] + results['total']['micro_recall'])
final_results['Original'] = results

pprint.pprint(final_results)


# In[12]:


embedded_text_input = Input(shape=(150, 300), name='embedded_text_input')
filters_fe = [Conv1D(f[0], f[1], activation='relu', weights = model.layers[2+idx].get_weights(), trainable = False)(embedded_text_input) for idx, f in enumerate(filters)]
max_pools_fe = [GlobalMaxPool1D()(result) for result in filters_fe]
concatenated_fe = concatenate(max_pools_fe, axis=-1)

feature_extraction_model = Model(embedded_text_input, concatenated_fe)
feature_extraction_model.summary()


analyzer = innvestigate.create_analyzer('lrp.epsilon', feature_extraction_model, neuron_selection_mode="index")

embeddings_func = K.function([model.layers[0].input],[model.layers[1].output])
input = [X_train[:min(2000, len(X_train))]]
num_total = len(input[0])
num_batches = math.ceil(num_total / 128)
X_train_emb = None
for b in tqdm(range(num_batches)):
    this_input = [i[128*b: min(len(i), 128*(b+1))] for i in input]
    this_output = embeddings_func(this_input)
    if X_train_emb is None:
        X_train_emb = this_output
    else:
        for i in range(len(X_train_emb)):
            X_train_emb[i] = np.concatenate((X_train_emb[i], this_output[i]), axis = 0)
X_train_emb = X_train_emb[0]

def get_window_size_from_feature_index(feature_idx, filters):
    assert feature_idx >= 0 and feature_idx < 30
    cumulative_idx = 0
    for p in filters:
        cumulative_idx += p[0]
        if feature_idx < cumulative_idx:
            return p[1]

top_ngrams_of_features = {}
ngrams_stat_of_features = [[] for i in range(30)]
for feature_idx in tqdm(range(30)):
    top_ngrams_of_features[feature_idx] = {}
    relevance_scores = np.array([]).reshape(0, len(X_train[0]))
    i = 0
    while i < len(X_train_emb):
        relevance_scores = np.vstack([relevance_scores, analyzer.analyze(X_train_emb[i:min(i+100, len(X_train_emb))], feature_idx).sum(axis = -1)])
        i += 100
    for example_idx, relevance_vector in enumerate(relevance_scores):
        score = sum(relevance_vector)
        ngram = ' '.join([index2word[i] if i != 0 else '_' for i in X_train[example_idx][relevance_vector.nonzero()]])
        if score in top_ngrams_of_features[feature_idx]:
            top_ngrams_of_features[feature_idx][score].append(ngram)
        else:
            top_ngrams_of_features[feature_idx][score] = [ngram]

    for score, ngrams_list in top_ngrams_of_features[feature_idx].items():
        expected_space = get_window_size_from_feature_index(feature_idx, filters) - 1
        count = len(ngrams_list)
        if ngrams_list[0].count(' ') <= expected_space:
            rep = ngrams_list[0]
        else:
            rep = ' '.join(ngrams_list[0].split(' ')[0:expected_space+1])
        ngrams_stat_of_features[feature_idx].append({"ngram": rep, "score": score, "count": count})
    ngrams_stat_of_features[feature_idx].sort(key=lambda x: x["score"], reverse = True)

is_feature_enabled = [False]*30


# In[15]:


glove = api.load("glove-wiki-gigaword-300")


# In[16]:


similarity_index = WordEmbeddingSimilarityIndex(glove)

def get_final_scores(class_name, ngrams_stats):
    final_scores, max_similarity, max_score, max_count = [], -1, -1, -1
    for ngrams_stat in ngrams_stats:
        vector1 = simple_preprocess(ngrams_stat["ngram"], min_len=0, max_len=float("inf"))
        vector2 = simple_preprocess(class_name, min_len=0, max_len=float("inf"))
        dictionary = Dictionary([vector1, vector2])
        tfidf = TfidfModel(dictionary=dictionary)
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
        vector1_tf = tfidf[dictionary.doc2bow(vector1)]
        index = SoftCosineSimilarity(tfidf[dictionary.doc2bow(vector2)], similarity_matrix)
        try:
            similarity = index[vector1_tf]
        except:
            if class_name in ngrams_stat["ngram"]: similarity = 1
            else: similarity = 0
        final_score = (similarity, ngrams_stat["score"], ngrams_stat["count"])
        if similarity > max_similarity: max_similarity = similarity
        if ngrams_stat["score"] > max_score: max_score = ngrams_stat["score"]
        if ngrams_stat["count"] > max_count: max_count = ngrams_stat["count"]
        final_scores.append(final_score)
    new_final_scores = [((final_score[0]/max_similarity)+(final_score[1]/max_score)+(final_score[2]/max_count)) for final_score in final_scores]
    return sum(new_final_scores)


# In[17]:


feature_scores = []
for feature_idx in range(30):
    print(f"Feature number: {feature_idx+1}")
    W = model.layers[-1].get_weights()[0]
    scores = {}
    for idx, cn in enumerate(class_names):
        scores[cn] = W[feature_idx][idx]
    max_score, max_class_name = float("-inf"), ""
    for idx, cn in enumerate(class_names):
        if scores[cn] > max_score:
            max_score = scores[cn]
            max_class_name = cn

    final_score = get_final_scores(max_class_name, ngrams_stat_of_features[feature_idx])

    feature_scores.append((feature_idx, final_score))


# In[21]:


feature_scores.sort(key = lambda x: x[1], reverse=True)
max_value, num_features = 0, None

y_validate = data["y_validate"]

for i in range(1, 31):
    new_feature_scores = feature_scores[i:]
    for feature_score in new_feature_scores: is_feature_enabled[feature_score[0]] = True

    text_input = Input(shape=(None,), dtype="int32")
    embedded_text = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=150, trainable=True)(text_input)

    filter_layers = [Conv1D(f[0], f[1], activation='relu', trainable=True)(embedded_text) for f in filters]
    max_pool_layers = [GlobalMaxPool1D()(result) for result in filter_layers]

    concatenated = concatenate(max_pool_layers,axis=-1)

    ans = MaskedDense(len(class_names), activation='softmax')(concatenated)

    model_improved = Model(text_input, ans)
    model_improved.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_improved.summary()
    model_improved.set_weights(model.get_weights())

    for idx, enable in enumerate(is_feature_enabled):
        if not enable:
            model_improved.layers[-1].disable_mask(idx)

    y_train_onehot, y_validate_onehot = to_categorical(data['y_train']), to_categorical(data['y_validate'])

    checkpointer = ModelCheckpoint(filepath="trained_CNN.h5", verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)

    history = model_improved.fit(X_train, y_train_onehot, verbose = 2, epochs=300, batch_size=128, callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))

    accuracy = history.history["val_acc"][-1]

    if accuracy > max_value:
        max_value = accuracy
        num_features = i

final_results["features"] = num_features
new_feature_scores = feature_scores[:num_features]
for feature_score in new_feature_scores: is_feature_enabled[feature_score[0]] = True


# In[16]:


text_input = Input(shape=(None,), dtype="int32")
embedded_text = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=150, trainable=True)(text_input)

filter_layers = [Conv1D(f[0], f[1], activation='relu', trainable=True)(embedded_text) for f in filters]
max_pool_layers = [GlobalMaxPool1D()(result) for result in filter_layers]

concatenated = concatenate(max_pool_layers,axis=-1)

ans = MaskedDense(len(class_names), activation='softmax')(concatenated)

model_improved = Model(text_input, ans)
model_improved.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_improved.summary()
model_improved.set_weights(model.get_weights())

for idx, enable in enumerate(is_feature_enabled):
    if not enable:
        model_improved.layers[-1].disable_mask(idx)

y_train_onehot, y_validate_onehot = to_categorical(data['y_train']), to_categorical(data['y_validate'])

checkpointer = ModelCheckpoint(filepath="trained_CNN.h5", verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

model_improved.fit(X_train, y_train_onehot, verbose = 2, epochs=300, batch_size=128, callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))


# In[23]:


prediction_test_onehot = model_improved.predict(X_test, batch_size=128)
prediction_test = prediction_test_onehot.argmax(axis=1).squeeze()
results = {'per_class': {}, 'total': {}}
y_true, y_predict = np.array(y_test), np.array(prediction_test)
for idx, cname in enumerate(class_names):
    TP = np.sum(np.logical_and((y_true == y_predict), (y_true == idx)))
    class_precision = TP / np.sum(y_predict == idx)
    class_recall = TP / np.sum(y_true == idx)
    class_f1 = 2*class_precision*class_recall / (class_precision + class_recall)
    results['per_class'][idx] = {'class_name': cname,
                                 'true_positive': TP,
                                 'all_positive': np.sum(y_predict == idx),
                                 'all_true': np.sum(y_true == idx),
                                 'class_precision': class_precision,
                                 'class_recall': class_recall,
                                 'class_f1': class_f1}

results['total']['macro_precision'] = np.mean([results['per_class'][idx]['class_precision'] for idx in results['per_class']])
results['total']['macro_recall'] = np.mean([results['per_class'][idx]['class_recall'] for idx in results['per_class']])
results['total']['macro_f1'] = 2 * results['total']['macro_precision'] * results['total']['macro_recall'] / (results['total']['macro_precision'] + results['total']['macro_recall'])
results['total']['accuracy'] = sum([results['per_class'][idx]['true_positive'] for idx in results['per_class']]) / len(y_true)
results['total']['micro_precision'] = np.sum([results['per_class'][idx]['true_positive'] for idx in results['per_class']]) / np.sum([results['per_class'][idx]['all_positive'] for idx in results['per_class']])
results['total']['micro_recall'] = np.sum([results['per_class'][idx]['true_positive'] for idx in results['per_class']]) / np.sum([results['per_class'][idx]['all_true'] for idx in results['per_class']])
results['total']['micro_f1'] = 2 * results['total']['micro_precision'] * results['total']['micro_recall'] / (results['total']['micro_precision'] + results['total']['micro_recall'])
final_results['Mine'] = results

pprint.pprint(final_results)

json.dump(final_results, open(output_filename, 'w'), cls=NpEncoder)
