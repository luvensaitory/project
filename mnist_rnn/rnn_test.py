# -*- coding: utf-8 -*-
# 導入函式庫
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from PIL import Image
# 固定亂數種子，使每次執行產生的亂數都一樣
np.random.seed(1337)


# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 將 training 的 input 資料轉為3維，並 normalize 把顏色控制在 0 ~ 1 之間
X_train = X_train.reshape(-1, 28, 28) / 255.      
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = load_model('SimpleRNN.h5')
model.load_weights("SimpleRNN.weight")
img1 = Image.open('01.jpg')
img1 = img1.convert("L")
#img1 = np.fromstring(img1.tobytes(), dtype=np.uint8)
img1 = img1.resize((28, 28))
img1 = np.asarray(img1)
img1 = img1.reshape(-1, 28, 28)/255.
# 一批訓練多少張圖片

# 預測(prediction)
#X = X_test[0:10,:]
predictions = model.predict_classes(img1)
# get prediction result
print(predictions)
