import redisAI as rai
# from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np

vectorizer = pickle.load(open('/root/dev/projects/redisai-non-tensor-model/countVectorizer.pickle', 'rb'))

def runModel(x):
    print(x['text'])
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
    res = execute('XADD', 'xOut', 'MAXLEN', '~', 1000, '*', 'streamId', x['streamId'], 'text', x['text'], 'pred', x['output_label'])
    print(str(res))
    
bg = GearsBuilder('StreamReader')
bg.foreach(lambda x: print(str(x['streamId'])))
bg.foreach(runModel)
# bg.foreach(storeResults)
bg.register('xIn')