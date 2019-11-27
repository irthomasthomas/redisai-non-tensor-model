import redisAI as rai
# from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import redis

conn = redis.Redis(host="localhost", port=6379, db=0)
print(str(conn))
vectorizer = pickle.load(open('/root/dev/projects/redisai-non-tensor-model/countVectorizer.pickle', 'rb'))

def runModel(x):
    sample = vectorizer.transform(
        [x['text']]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    modelRunner = rai.createModelRunner('sklmodel')
    rai.modelRunnerAddInput(modelRunner, 'float_input', ba)
    rai.modelRunnerAddOutput(modelRunner, 'output_label')
    rai.modelRunnerAddOutput(modelRunner, 'output_probability')
    model_replies = rai.modelRunnerRun(modelRunner)
        # ERROR: type 0 is not supported in this function
    # model_output = model_replies[0]
    print("runModel output...")
    print(str(model_replies)) # = None

def storeResults(x):
    print(str(x))
    res = execute(
        'XADD', 'xOut', 'MAXLEN', '~', 1000, '*',
        'streamId', x['streamId'], 'text', x['text'],
        'pred', x['output_label'])
    print(str(res))
    
def runModel2(x):
    sample = vectorizer.transform(
        [x['text']]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    conn.execute_command(
        'AI.TENSORSET', 'tensor', 'FLOAT', '1', '20', 'BLOB', ba.tobytes())
    conn.execute_command(
        'AI.MODELRUN', 'sklmodel', 'INPUTS', 'tensor', 'OUTPUTS', 'out_label', 'out_probs')
    out = conn.execute_command(
        'AI.TENSORGET', 'out_label', 'VALUES')
    print(out[2])


# TODO: Run model through normal redis client
bg = GearsBuilder('StreamReader')
bg.foreach(runModel)
# bg.foreach(storeResults)
bg.register('xIn')

