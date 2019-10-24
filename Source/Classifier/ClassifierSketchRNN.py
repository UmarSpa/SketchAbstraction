from .modelSketchRNN import Classifier as myClassifier
import tensorflow as tf
import numpy as np

#####################################################
#   Model Initialization
#####################################################
def model_initialization(modelName="Classifier"):
    with tf.variable_scope(modelName):
        Classifier = myClassifier()
    return Classifier

#####################################################
#   Model Loading
#####################################################
def load_model_(model_file):
    d = np.load(model_file, allow_pickle=True).item()
    init_ops = []
    model_variables = [var for var in tf.trainable_variables() if "Classifier" in var.name]
    print('\n** Loading Classifier Weights **')
    for var in model_variables:
        varName = var.name
        init_ops.append(var.assign(d[varName]))
        print (varName)
    print('** Loading Classifier Weights - Complete **\n')
    return init_ops

def model_loading(sess, modelDir='./Source/Classifier/Weights/sketchRNN.npy'):
    init_Class = load_model_(modelDir)
    sess.run(init_Class)

#####################################################
#   Model sampling
#####################################################
def model_pred(sess, Classifier, batch_x, batch_x_l):
    feed_dict = {Classifier.input_data: batch_x, Classifier.input_data_lens: batch_x_l}
    preds, values, indices = sess.run([Classifier.preds, Classifier.values, Classifier.indices], feed_dict=feed_dict)
    return preds[0], values[0], indices[0]
