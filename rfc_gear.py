import redisAI as rai
# from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np


def runModel(x):
    vectorizer = pickle.load(open('/root/dev/projects/redisai-non-tensor-model/countVectorizer.pickle', 'rb'))

    text_sample = vectorizer.transform(
        [x['text']]).toarray()

    modelRunner = rai.createModelRunner('sklmodel')

    ba = np.asarray(text_sample, dtype=np.float32)
    ba = bytearray(ba.tobytes())

    tensor = rai.createTensorFromBlob('FLOAT', [1, 10], ba)
    rai.modelRunnerAddInput(modelRunner, 'float_input', tensor)
    rai.modelRunnerAddOutput(modelRunner, 'output_label')
    rai.modelRunnerAddOutput(modelRunner, 'output_probability')
    print(x['text'])

    # SEG FAULTS HERE
    # model_replies = rai.modelRunnerRun(modelRunner)

    model_output = model_replies[0]
    print("runModel output...")
    print(model_output)

def storeResults(x):
    print(str(x))
    res = execute('XADD', 'xOut', 'MAXLEN', '~', 1000, '*', 'streamId', x['streamId'], 'text', x['text'], 'pred', x['output_label'])
    print(str(res))
    
bg = GearsBuilder('StreamReader')
bg.foreach(lambda x: print(str(x['streamId'])))
bg.foreach(runModel)
bg.foreach(storeResults)
bg.register('xIn')