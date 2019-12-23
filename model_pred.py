from tensorflow import keras
import jieba.posseg as pseg
import pandas as pd
import numpy as np
from sklearn.externals import joblib

def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x' and flag!='y'and word!='丶'] )


model=keras.models.load_model('lstm.h5')

print(model.summary())



pos_test_filename = 'source/normal_test.txt'
neg_test_filename = 'source/die_test.txt'

col_name=['content']

pos_test_df =  pd.read_table(pos_test_filename,names=col_name)

neg_test_df =  pd.read_table(neg_test_filename,names=col_name)


pos_test_df['content_tokenized'] = \
    pos_test_df.loc[:, 'content'] \
         .apply(jieba_tokenizer)

neg_test_df['content_tokenized'] = \
    neg_test_df.loc[:, 'content'] \
         .apply(jieba_tokenizer)

test_df=pd.concat([pos_test_df,neg_test_df],ignore_index=True)
#print(test_df)
#MAX_NUM_WORDS = 10000
tokenizer = joblib.load('tk.pkl')

x1_test = tokenizer \
    .texts_to_sequences(
        pos_test_df.content_tokenized)
x2_test = tokenizer \
    .texts_to_sequences(
        neg_test_df.content_tokenized)



MAX_SEQUENCE_LENGTH=3655

x1_test = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(
        x1_test, 
        maxlen=MAX_SEQUENCE_LENGTH)


x2_test = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(
        x2_test, 
        maxlen=MAX_SEQUENCE_LENGTH)

x_test=np.vstack((x1_test,x2_test))


#print(x_test)




predictions = model.predict(
    x_test)
'''
for i in predictions:
    print(np.argmax(i))

'''
test_df['label']=np.argmax(predictions, axis=1)

for index, row in test_df.iterrows():
    print('index:{},content:{},label:{}'.format(index,row['content'],row['label']))
    
