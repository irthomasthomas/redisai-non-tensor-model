import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import redis
import redisai
import pickle
from gearsclient import GearsRemoteBuilder as GearsBuilder


corpus = [
    "The apple is on sale apple", "The orange are on sale",
    "The apple is present", "The orange is present",
    "the best fruit is apple", "best fruit is orange",
    "apple is my favourite fruit", "orange are my favourite",
    "apple are the best", "orange are the best",
    "an apple a day keeps the grocer in pay", "orange a day keeps the doc away"]

Y = np.array([0,1,0,1,0,1,0,1,0,1,0,1]) # segfault on apple and orange
# Y = np.array([1,0,1,0,1,0,1,0,1,0,1,0]) # segfault on apple
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus).toarray()
pickle.dump(vectorizer, open("/root/dev/projects/redisai-non-tensor-model/countVectorizer.pickle", "wb"))

classifier = RandomForestClassifier()
classifier.fit(X, Y)

sample_text = "apple is present"
print(sample_text)
sample = vectorizer.transform(
        [sample_text]).toarray()

pred = classifier.predict(sample)
print("rfc model prediction...")
print(pred)
print("SHAPE")
print(X.shape) # 12, 22
########################################

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [("float_input", FloatTensorType([1, 20]))]
onx = convert_sklearn(classifier, initial_types=initial_type)
with open("rfc_onx.onnx", "wb") as f:
        f.write(onx.SerializeToString())
print("saved onx")
import onnxruntime as rt
sess = rt.InferenceSession("rfc_onx.onnx")

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: sample.astype(np.float32)})[0]
print("onnx prediction...")
print(pred_onx)

########################################

conn = redis.Redis(host="localhost", port=6379, db=0)
with open("rfc_onx.onnx", "rb") as f:
    model = f.read()
    res = conn.execute_command('AI.MODELSET', 'sklmodel', 'ONNX', 'CPU', model)
rai = redisai.Client()
rai.tensorset('tensor', sample.astype(np.float32))
rai.modelrun(
    'sklmodel', inputs=['tensor'], outputs=['out_label', 'out_probs'])
out = rai.tensorget('out_label')
print("RedisAI prediction...")
print(out)

#########################################

ba = np.asarray(sample, dtype=np.float32)
conn.execute_command(
    'AI.TENSORSET', 'tensor', 'FLOAT', '1', '20', 'BLOB', ba.tobytes())
conn.execute_command('AI.MODELRUN', 'sklmodel', 'INPUTS', 'tensor', 'OUTPUTS', 'out_label', 'out_probs')
out = conn.execute_command('AI.TENSORGET', 'out_label', 'VALUES')
print(out[2])

##########################################

initialized_key = '{}:initialized'.format('xIn')
# Check if this Redis instance had already been initialized
initialized = conn.exists(initialized_key)
if initialized:
    print('Discovered evidence of a privious initialization - skipping.')
    exit(0)
print("Load gear")
with open('rfc_gear.py', 'rb') as f:
    gear = f.read()
    res = conn.execute_command('RG.PYEXECUTE', gear)
    print(res)
print('Flag initialization as done - ', end='') 
print(conn.set(initialized_key, 'most certainly.'))