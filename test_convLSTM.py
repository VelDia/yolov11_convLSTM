import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Masking, ConvLSTM2D, LSTM, Bidirectional, Dense
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization

batch_size=68
units=128
learning_rate=0.005
epochs=20
dropout=0.2
recurrent_dropout=0.2
X_train = np.random.rand(700, 50,34)
y_train = np.random.choice([0, 1], 700)
X_test = np.random.rand(100, 50, 34)
y_test = np.random.choice([0, 1], 100)
loss = tf.losses.binary_crossentropy

model = tf.keras.models.Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
# uncomment the line beneath for convolution
# model.add(Conv1D(filters=32, kernel_size=8, strides=1, activation="relu", padding="same"))
model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
model.add(BatchNormalization())

model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(loss=loss,
              optimizer=adamopt,
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    verbose=1)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)

yhat = model.predict(X_test)