# suicide_lstm
#檔案說明:

1.source:自殺留言資料集

2.model_train.py:用自殺留言資料集進行lstm模型訓練

3.model_pred.py:用自殺留言lstm模型進行測試資料的預測

4.suicide_model.ipynb:原本的model_train與model_pred的輸出是採用二分法的分類方式，不過這個版本是採用一維輸出，還有進行一些模型的微調。

模型介紹:

1.開源工具:tensorflow,jieba,keras,pandas

2.模型類型:LSTM

3.模型相關說明:

![image](https://github.com/openaifab/suicide_lstm/blob/master/model_detail.png)

4.模型input資料的前處理:對中文留言做分詞並做停用詞的過濾

5.模型訓練資料:900筆簡體中文的留言(450筆自殺留言，450筆非自殺留言)

6.模型input:文字類型，沒有字數限制，會做繁體轉簡體的動作

7.模型output:介於0~100%的自殺風險評估百分比分數

8.模型訓練時間:五分鐘之內

9.模型預測時間:五分鐘之內(預測一千筆留言)


![image](https://github.com/openaifab/suicide_lstm/blob/master/suicide_lstm.jpg)
