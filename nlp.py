import numpy as np
import re
import os
import pickle
from nltk.corpus import stopwords
from random import shuffle
from tqdm import *
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM, Conv1D, MaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

MODEL_NAME = "nlp.h5"

def process_data():
    file = open('training.txt', 'r')
    lines = file.readlines()

    train_data = []
    train_label = []
    for line in lines:
        train_label.append(int(line[:1]))
        train_data.append(re.sub('[^A-Za-z]+', ' ', line[2:].lower()))
    train_label = to_categorical(train_label, 2)

    file = open('testdata.txt', 'r')
    lines = file.readlines()

    test_data = []
    for line in lines:
        test_data.append(re.sub('[^A-Za-z]+', ' ', line.lower()))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words for words in train_data if words not in stopwords.words('english'))
    train_voc_max_length = max(len(lines.split()) for lines in train_data)
    train_voc_size = len(tokenizer.word_index) + 1
    train_voc = tokenizer.texts_to_sequences(train_data)
    train_voc = pad_sequences(train_voc, maxlen=train_voc_max_length, padding='post')

    test_voc_max_length = max(len(lines.split()) for lines in test_data)
    test_voc_size = len(tokenizer.word_index) + 1
    test_voc = tokenizer.texts_to_sequences(test_data)
    test_voc = pad_sequences(test_voc, maxlen=train_voc_max_length, padding='post')

    training_data = []
    for i, data in tqdm(enumerate(train_voc)):
        label = train_label[i]
        training_data.append([np.array(data), np.array(label)])
    shuffle(training_data)

    testing_data = []
    for i, data in tqdm(enumerate(test_voc)):
        testing_data.append([np.array(data), i+1])

    return training_data, testing_data, train_voc_max_length, test_voc_size, tokenizer

def train():
    training_data, testing_data, train_voc_max_length, test_voc_size, tokenizer = process_data()

    model = Sequential()
    model.add(Embedding(test_voc_size, 64, input_length=train_voc_max_length))

    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(MODEL_NAME):
        model.load_weights(MODEL_NAME)
        print 'model exists'

    train = training_data[:int(len(training_data) * 0.9)]
    test = training_data[int(len(training_data) * 0.9):]
    X = np.array([i[0] for i in train])
    y = np.array([i[1] for i in train])

    test_x = np.array([i[0] for i in test])
    test_y = np.array([i[1] for i in test])

    # model.fit(X, y, epochs=5, batch_size=32, verbose=1, validation_data=(test_x, test_y))
    #
    # model.save(MODEL_NAME)

    return model, testing_data, train_voc_max_length, tokenizer

def test(model, testing_data, train_voc_max_length, tokenizer):
    input = raw_input("Input ")
    cleaned = [re.sub('[^A-Za-z]+', ' ', input.lower())]

    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

    sentence = tokenizer.texts_to_sequences(cleaned)
    sentence = pad_sequences(sentence, maxlen=train_voc_max_length, padding='post')

    model_out = model.predict([sentence])[0]
    label = np.argmax(model_out)
    print label

    # with open('submissionfile.txt', 'w') as f:
    #     for i, data in tqdm(enumerate(testing_data)):
    #         num = data[1]
    #         sentence = data[0]
    #
    #         model_out = model.predict(np.array([sentence]))[0]
    #
    #         label = np.argmax(model_out)
    #         f.write('{}\n'.format(label))

if __name__ == "__main__":
    model, testing_data, train_voc_max_length, tokenizer = train()
    test(model, testing_data, train_voc_max_length, tokenizer)


















