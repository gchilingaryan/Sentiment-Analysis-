import numpy as np
import pickle
import re
import tensorflow as tf
from keras import models
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify

global graph
graph = tf.get_default_graph()
model = models.load_model('../nlp.h5')
tokenizer = pickle.load(open('../tokenizer.pkl', 'rb'))

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('html/index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text', None)
    assert text is not None

    with graph.as_default():
        cleaned = [re.sub('[^A-Za-z]+', ' ', text.lower())]

        sentence = tokenizer.texts_to_sequences(cleaned)
        sentence = pad_sequences(sentence, maxlen=42, padding='post')

        model_out = model.predict([sentence])[0]
        label = np.argmax(model_out)

        s = 'Positive' if label == 1 else 'Negative'
        p = str(max(model_out))
        return jsonify({
            'sentiment': s,
            'probability': p
        })

app.run()