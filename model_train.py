import pandas as pd

import jieba.posseg as pseg

from tensorflow import keras

import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
 
import os

from sklearn.model_selection \
    import train_test_split

from sklearn.externals import joblib


def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x' and flag!='y'and word!='丶'] )




pos_train_filename = 'source/normal.txt'
neg_train_filename = 'source/die.txt'


col_name=['content']

pos_train_df =  pd.read_table(pos_train_filename,names=col_name)

neg_train_df =  pd.read_table(neg_train_filename,names=col_name)




#print(pos_train_df)

#print(neg_train_df)

pos_train_df['content_tokenized'] = \
    pos_train_df.loc[:, 'content'] \
         .apply(jieba_tokenizer)

neg_train_df['content_tokenized'] = \
    neg_train_df.loc[:, 'content'] \
         .apply(jieba_tokenizer)






pos_train_df['label'] = 0
    

neg_train_df['label'] = 1

#print(pos_train_df)

#print(neg_train_df)




y_train=pd.concat([pos_train_df['label'],neg_train_df['label']])

y_train = keras \
    .utils \
    .to_categorical(y_train)


#print(label_train)


#print(type(pos_train_df['label']))

#print(label_train['label'])

MAX_NUM_WORDS = 10000
tokenizer = keras \
    .preprocessing \
    .text \
    .Tokenizer(num_words=MAX_NUM_WORDS)



corpus_x1 = pos_train_df.content_tokenized
corpus_x2 = neg_train_df.content_tokenized
corpus = pd.concat([
    corpus_x1, corpus_x2],ignore_index=True)


#print(corpus.shape)
#print(corpus)


tokenizer.fit_on_texts(corpus)

joblib.dump(tokenizer,'tk.pkl')


x1_train = tokenizer \
    .texts_to_sequences(corpus_x1)
x2_train = tokenizer \
    .texts_to_sequences(corpus_x2)

max_seq_len = max([
    len(seq) for seq in x1_train])
print(max_seq_len)

#os._exit() 

MAX_SEQUENCE_LENGTH = max_seq_len
x1_train = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(x1_train, 
                   maxlen=MAX_SEQUENCE_LENGTH)

x2_train = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(x2_train, 
                   maxlen=MAX_SEQUENCE_LENGTH)



x_train=np.vstack((x1_train,x2_train))


#print(x_train)


VALIDATION_RATIO = 0.1

RANDOM_STATE = 9527

x_train, x_val, \
y_train, y_val = \
    train_test_split(
        x_train, y_train, 
        test_size=VALIDATION_RATIO, 
        random_state=RANDOM_STATE
)




print("Training Set")
print("-" * 10)
print(f"x_train: {x_train.shape}")

print(f"y_train : {y_train.shape}")

print("-" * 10)
print(f"x_val:   {x_val.shape}")

print(f"y_val :   {y_val.shape}")
print("-" * 10)
print("Test Set")

# 基本參數設置，有幾個分類
NUM_CLASSES = 2



# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256

# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128

model = Sequential()
model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
model.add(Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM))
model.add(LSTM(NUM_LSTM_UNITS))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print(model.summary())


# 決定一次要放多少成對標題給模型訓練
BATCH_SIZE = 32

# 決定模型要看整個訓練資料集幾遍
NUM_EPOCHS = 10

# 實際訓練模型
history = model.fit(
    # 輸入是兩個長度為 20 的數字序列
    x=x_train, 
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    # 每個 epoch 完後計算驗證資料集
    # 上的 Loss 以及準確度
    validation_data=(
        x_val, 
        y_val
    ),
    # 每個 epoch 隨機調整訓練資料集
    # 裡頭的數據以讓訓練過程更穩定
    shuffle=True
)


model.save('lstm.h5')


print('finished')
'''
text='我丶不配丶结束饭。我好累啊。饭。'

words = pseg.cut(text)
words=[word for word in words]
print(words)
'''
