model = tf.keras.models.Sequential()
    
model.add(Conv1D(filters=32, kernel_size=8, strides=1, activation="relu", padding="same",input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(filters=16, kernel_size=8, strides=1, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size = 2))


model.add(Masking(mask_value=0.0))
model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))