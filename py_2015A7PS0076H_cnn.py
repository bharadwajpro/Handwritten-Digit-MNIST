from keras.datasets import mnist
from keras.layers import Dense
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping

img_rows = 28
img_cols = 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(50, (5, 5), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D((5, 5)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(min_delta=10e-6, patience=20)
history = model.fit(x_train, y_train, batch_size=128, epochs=4000, validation_split=0.3, callbacks=[early_stopping])
score = model.evaluate(x_test, y_test)
print(model.summary())
print('Train loss:', history.history['loss'][-1])
print('Train accuracy:', history.history['acc'][-1])
print('Validation loss:', history.history['val_loss'][-1])
print('Validation accuracy:', history.history['val_acc'][-1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
