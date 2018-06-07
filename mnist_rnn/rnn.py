import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

a = np.zeros((70, 7))

for i in range(70):
    for j in range(7):
        if np.random.rand()>0.5:
            a[i][j] += 1

b = np.random.rand(70, 2)

data_array = np.hstack((a, b))
data_array = data_array[:, np.newaxis, :]
class_array = np.zeros((70, 6))
class_array_temp = np.random.randint(6, size=70)
for i in range(70):
    class_array[i][class_array_temp[i]] = 1
# 70 data, 60 train, 10 test
X_train = data_array[0:60, :, :]
X_test = data_array[60:70, :, :]
y_train = class_array[0:60, :]
y_test = class_array[60:70, :]

#build model
model = Sequential()

model.add(SimpleRNN(batch_input_shape=(None, 1, 9), units = 50))
model.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
'''
for step in range(1, 51):
    
    loss = model.train_on_batch(X_train, y_train)
    
    # 每 500 批，顯示測試的準確率
    # 模型評估
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], 
        verbose=False)
    print("test loss: {}  test accuracy: {}".format(loss,accuracy))
'''
X = X_test
predictions = model.predict_classes(X)
# get prediction result

accuracy = 0

for i in range(10):
    for j in range(6):
        if y_test[i][j] == 1:
            ans = j
    if ans == predictions[i]:
        accuracy += 1

print(accuracy/10)