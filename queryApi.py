#This file queries for data
import numpy as np
import requests
import codecs, json 
X= np.load("X.npy",allow_pickle=True)
#independent_var = np.expand_dims(X[0], axis=0)

url = 'http://127.0.0.1:5000'
val = X[0].tolist() 
file_path = "/X.json" 
data=json.dumps(val)
print(X[0].shape)

print("Wrapped in json format")
req=requests.post(url, data)
print(req)
