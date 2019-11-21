import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import redis
import redisai
import pickle
from gearsclient import GearsRemoteBuilder as GearsBuilder


corpus = [
    "The apple is on sale apple", "The oranges are on orange sale",
    "The apple and is present", "The orange and is present"]

Y = np.array([0,1,0,1])
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus).toarray()
pickle.dump(vectorizer, open("countVectorizer.pickle", "wb"))

classifier = RandomForestClassifier()
classifier.fit(X, Y)

sample_text = "orange is present"
print(sample_text)
sample = vectorizer.transform(
        [sample_text]).toarray()

pred = classifier.predict(sample)
print("rfc model prediction...")
print(pred)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [("float_input", FloatTensorType([1, 10]))]
onx = convert_sklearn(classifier, initial_types=initial_type)
with open("rfc_onx.onnx", "wb") as f:
        f.write(onx.SerializeToString())

import onnxruntime as rt
sess = rt.InferenceSession("rfc_onx.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: sample.astype(np.float32)})[0]
print("onnx prediction...")
print(pred_onx)

conn = redis.Redis(host="localhost", port=6379, db=0)
with open("rfc_onx.onnx", "rb") as f:
    model = f.read()
    res = conn.execute_command('AI.MODELSET', 'sklmodel', 'ONNX', 'CPU', model)

rai = redisai.Client()
tensor = redisai.BlobTensor.from_numpy(
    sample.astype(np.float32))
rai.tensorset('tensor', tensor)
rai.modelrun('sklmodel', inputs=['tensor'], outputs=['out_label', 'out_probs'])
out = rai.tensorget('out_label')
print("RedisAI prediction...")
print(out)

print("Load gear")
with open('rfc_gear.py', 'rb') as f:
    gear = f.read()
    res = conn.execute_command('RG.PYEXECUTE', gear)
    print(res)