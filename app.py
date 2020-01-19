import pandas as pd
from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
import codecs, json
from tensorflow.python.keras.backend import set_session
import tensorflow as tf


def init():
	sess = tf.InteractiveSession()
	model = load_model('modelcolab.h5')
	print(model.summary())
	graph = tf.get_default_graph()
	return model, sess, graph

global model, sess, graph
model, sess, graph = init()
# load model
# app
app = Flask(__name__)


# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
	data = request.get_json(force=True)
	#obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
	#b_new = json.loads(data)
	a_new = np.array(data)
	print("unwrapped json")
	print(model)
	independent_var = np.expand_dims(a_new, axis=0)
	print(independent_var.shape)
	with sess.as_default():
		with graph.as_default():
			output = model.predict(independent_var).tolist()
	print(">>>>>>>>>>> ", output)
	return jsonify(output)
	


if __name__ == '__main__':
    app.run(port = 5000, debug=True)
