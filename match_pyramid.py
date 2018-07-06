# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:08:24 2018

@author: Wanyu Du
"""

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, BatchNormalization, Activation
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import dot
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import regularizers

# load dataset
def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(t_)
    return np.asarray(ids)

def get_texts(file_path, question_path, mode=None):
    qes = pd.read_csv(question_path, sep='	', dtype=str)
    qes = qes.dropna()
    file = pd.read_csv(file_path, sep=' ', dtype=str)
    file = file.dropna()
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = {}
    for i in range(len(qes)):
        all_words[qes.iloc[i,0]]=qes.iloc[i,1]
    texts1 = []
    texts2 = []
    for t_ in id1s:
        texts1.append(all_words[t_].split(' '))
    for t_ in id2s:
        texts2.append(all_words[t_].split(' '))
    if mode=='test':
        return texts1, texts2
    else:
        labels=list(file['label'])
        labels=to_categorical(np.array(labels), num_classes=2)
#        label_smooth=0.1
#        labels=labels.clip(label_smooth/2., 1.-label_smooth)
        return labels, texts1, texts2



fit_max_len=50
embed_size=50
num_conv2d_layers=1
filters_2d=[16,32]
kernel_size_2d=[[3,3], [3,3]]
mpool_size_2d=[[2,2], [2,2]]
dropout_rate=0.5
batch_size=32
  

TRAIN_PATH = 'data/qouraqp/relation_train.txt'
VALID_PATH = 'data/qouraqp/relation_valid.txt'
TEST_PATH = 'data/qouraqp/relation_test.txt'
QUESTION_PATH = 'data/qouraqp/corpus_preprocessed.txt'
WORD_EMBED='data/qouraqp/embed_glove_d50'
use_embed=False
model_path='checkpoints/mp_lrs.h5'
log_path='checkpoints/mp_lrs.txt'
sub_path='submission_mp_lrs.csv'

print('Load files...')
train_labels, train_texts1, train_texts2 = get_texts(TRAIN_PATH, QUESTION_PATH, 'train')
valid_labels, valid_texts1, valid_texts2 = get_texts(VALID_PATH, QUESTION_PATH, 'valid')
test_texts1, test_texts2 = get_texts(TEST_PATH, QUESTION_PATH, 'test')

print('Prepare word embedding...')
# pad the docs
padded_train_texts1=pad_sequences(train_texts1, maxlen=fit_max_len, padding='post')
padded_train_texts2=pad_sequences(train_texts2, maxlen=fit_max_len, padding='post')
padded_valid_texts1=pad_sequences(valid_texts1, maxlen=fit_max_len, padding='post')
padded_valid_texts2=pad_sequences(valid_texts2, maxlen=fit_max_len, padding='post')
padded_test_texts1=pad_sequences(test_texts1, maxlen=fit_max_len, padding='post')
padded_test_texts2=pad_sequences(test_texts2, maxlen=fit_max_len, padding='post')
# create a weight matrix for words in training docs
word_embedding = pd.read_csv(WORD_EMBED, header=None, sep=' ')
embedding_matrix=word_embedding.iloc[:,1:]
vocab_size=len(word_embedding)

print('Split train and valid set...')
train_labels, train_texts1, train_texts2 = train_labels, padded_train_texts1, padded_train_texts2
valid_labels, valid_texts1, valid_texts2 = valid_labels, padded_valid_texts1, padded_valid_texts2

print('Build model...')
query=Input(shape=(fit_max_len,), name='query')
doc=Input(shape=(fit_max_len,), name='doc')

if use_embed:
    q_embed=Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=True)(query)
    d_embed=Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=True)(doc)
else:
    q_embed=Embedding(vocab_size, embed_size, embeddings_initializer='uniform', trainable=True)(query)
    d_embed=Embedding(vocab_size, embed_size, embeddings_initializer='uniform', trainable=True)(doc)

layer1_dot=dot([q_embed, d_embed], axes=-1)
layer1_dot=Reshape((fit_max_len, fit_max_len, -1))(layer1_dot)
    
layer1_conv=Conv2D(filters=8, kernel_size=5, padding='same')(layer1_dot)
layer1_activation=Activation('relu')(layer1_conv)
z=MaxPooling2D(pool_size=(2,2))(layer1_activation)
    
for i in range(num_conv2d_layers):
    z=Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
    z=Activation('relu')(z)
    z=MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)
        
pool1_flat=Flatten()(z)
pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
mlp1=Dense(32)(pool1_flat_drop)
mlp1=Activation('relu')(mlp1)
out=Dense(2, activation='softmax')(mlp1)
    
model=Model(inputs=[query, doc], outputs=out)
model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# build dataset generator
def generator(texts1, texts2, labels, batch_size, min_index, max_index):
    i=min_index
    
    while True:
        if i+batch_size>=max_index:
            i=min_index
        rows=np.arange(i, min(i+batch_size, max_index))
        i+=batch_size
        
        samples1=texts1[rows]
        samples2=texts2[rows]
        targets=labels[rows]
        yield {'query':samples1, 'doc':samples2}, targets
        
def test_generator(texts1, texts2, batch_size, min_index, max_index):
    i=min_index
    
    while True:
        if i+batch_size>=max_index:
            i=min_index
        rows=np.arange(i, min(i+batch_size, max_index))
        i+=batch_size
        
        samples1=texts1[rows]
        samples2=texts2[rows]
        yield {'query':samples1, 'doc':samples2}
            
train_gen=generator(train_texts1, train_texts2, train_labels, batch_size=batch_size, min_index=0, max_index=len(train_texts1))
valid_gen=generator(valid_texts1, valid_texts2, valid_labels, batch_size=batch_size, min_index=0, max_index=len(valid_texts1))
test_gen=test_generator(padded_test_texts1, padded_test_texts2, batch_size=1, min_index=0, max_index=len(test_texts1))

print('Train classifier...')
history=model.fit_generator(train_gen, epochs=10, steps_per_epoch=len(train_texts1)//batch_size,
                  validation_data=valid_gen, validation_steps=len(valid_texts1)//batch_size, verbose=1,
                  callbacks=[ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True), 
#                             EarlyStopping(monitor='val_loss', patience=3), 
                             CSVLogger(log_path)])

#print('Predict...')
#model=load_model(model_path)
#preds=model.predict_generator(test_gen, steps=len(test_texts1))

print('Plot validation accuracy and loss...')
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(loss, label='acc')
plt.plot(val_loss, label='val_acc')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()