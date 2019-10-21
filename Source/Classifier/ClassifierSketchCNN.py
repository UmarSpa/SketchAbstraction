from .modelSketchCNN import Classifier as myClassifier
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
    d = np.load(model_file).item()
    init_ops = []
    model_variables = [var for var in tf.trainable_variables() if "Classifier" in var.name]
    print('\n** Loading Classifier Weights **')
    for var in model_variables:
        varName = var.name
        init_ops.append(var.assign(d[varName]))
        print (varName)
    print('** Loading Classifier Weights - Complete **\n')
    return init_ops

def model_loading(sess, modelDir='./Source/Classifier/Weights/sketchCNN.npy'):
    init_Class = load_model_(modelDir)
    sess.run(init_Class)

#####################################################
#   Model sampling
#####################################################
def model_pred(sess, Classifier, batch_x):
    feed_dict = {Classifier.input_data: batch_x, Classifier.dropoutFlag: False}
    pred_values, rank_values, rank_indices = sess.run([Classifier.fc8_preds, Classifier.values, Classifier.indices], feed_dict=feed_dict)
    return pred_values[0], rank_values[0], rank_indices[0]
