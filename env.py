import os
import math
import numpy as np
import tensorflow as tf
from Source.Classifier import ClassifierSketchRNN as ClassSketchRNN
from Source.Classifier import ClassifierSketchCNN as ClassSketchCNN
from utils import global_to_standard, sketch_raster, sketch_raster_info
from agent import Agent

class Env(object):
    def __init__(self, args):
        """Data loading and archiecture initialization."""
        self.args = args
        self.sketch_data_loading()
        self.initialization()

    def training(self):
        """Training of abstraction agent."""
        for trainIdx in range(self.args.trainEps):
            self.gen_sketch_sample()
            rewardAgent, gradBuff = self.sample_abstraction()
            rewardHuman, _ = self.sample_abstraction(inputType='Human')
            rewardRandom, _ = self.sample_abstraction(inputType='Random', abstractOrd=np.random.choice(self.sample.shape[0], self.sampleBudget, replace=False))
            rewardUpperBound = self.single_reward_sketch(self.sample.copy(), self.sampleLens.copy())

            rewardBuff = self.reward_elaboration(rewardAgent, rewardHuman, rewardRandom, self.annealingFac[trainIdx])
            self.gradientUpdate(gradBuff, rewardBuff)

            self.upperAccList.append(rewardUpperBound[0] == self.sampleClass)
            self.agentAccList.append(rewardAgent[-1][0] == self.sampleClass)
            self.rewardList.append(np.mean(rewardBuff))
            self.absLenList.append(self.sampleBudget)
            self.fullLenList.append(len(self.sample))

            if trainIdx % self.args.outFreq == 0 and trainIdx != 0: self.training_output(trainIdx)
            if trainIdx % self.args.testFreq == 0 and trainIdx != 0:
                self.testing(self.args.testStep)
                self.testAcc = np.mean(self.agentAccListTest)
                if self.testAcc > self.bestModelAcc: self.save_model()

    def testing(self, testIdxStep=1):
        """Testing of abstraction agent."""
        self.args.testIdxStep = testIdxStep
        self.args.trainFlag = False
        self.testIdx, self.testClass = 0, 0
        self.agentRNNDropout, self.agentFCDropout = 1.0, 1.0
        self.agentAccListTest, self.upperAccListTest, self.absLenListTest, self.fullLenListTest = ([] for _ in range(4))
        while not self.args.trainFlag:
            self.gen_sketch_sample()
            rewardAgent, _ = self.sample_abstraction()
            rewardUpperBound = self.single_reward_sketch(self.sample.copy(), self.sampleLens.copy())

            self.upperAccListTest.append(rewardUpperBound[0] == self.sampleClass)
            self.agentAccListTest.append(rewardAgent[-1][0] == self.sampleClass)
            self.absLenListTest.append(self.sampleBudget)
            self.fullLenListTest.append(len(self.sample))

        self.testing_output()

    def sample_abstraction(self, inputType='Agent', abstractOrd=[]):
        """Abstraction of the data sample under processing, along with the reward and gradient computation."""
        gradientBuff, rewardBuff = ([] for _ in range(2))
        candidateBucket, candidateBucketLens, candidateFeatBucket = self.sample.copy(), self.sampleLens.copy(), self.sampleFeat.copy()
        candidateFeatBucket = np.concatenate((candidateFeatBucket, np.array([np.eye(10)[sample_] for sample_ in self.sampleTimeStamp])), axis=1)
        chosenBucket = np.zeros((1, 101, 3), dtype='float32')
        chosenFeatBucket = np.zeros((1, 256 + 10), dtype='float32')
        self.startPos, self.normalizingFac = sketch_raster_info(candidateBucket.copy(), candidateBucketLens.copy())

        for absIdx in range(self.sampleBudget):
            if inputType=='Agent':
                feed_dict = {self.myAgent.chosen_strokes: np.expand_dims(chosenFeatBucket, axis=0),
                             self.myAgent.num_chosen_strokes: [chosenFeatBucket.shape[0]],
                             self.myAgent.rnn_dropout: self.agentRNNDropout,
                             self.myAgent.candidate_strokes: candidateFeatBucket,
                             self.myAgent.num_candidate_strokes: candidateFeatBucket.shape[0],
                             self.myAgent.sketch_class: np.repeat(np.eye(self.numClasses)[self.sampleClass].reshape(1, -1), candidateFeatBucket.shape[0], axis=0),
                             self.myAgent.full_dropout: self.agentFCDropout}
                if self.args.trainFlag:
                    action_, gradients = self.sess.run([self.myAgent.action, self.myAgent.gradients], feed_dict=feed_dict)
                    actionVal = action_[0, 0]
                else:
                    chosen_stroke = self.sess.run(self.myAgent.log_prob, feed_dict=feed_dict)
                    actionVal = np.argmax(chosen_stroke)
                    gradients = []
            elif inputType=='Human':
                actionVal = 0
                gradients = []
            elif inputType=='Random':
                if absIdx != 0: abstractOrd[abstractOrd[absIdx-1] < abstractOrd] = abstractOrd[abstractOrd[absIdx-1] < abstractOrd] - 1
                actionVal = abstractOrd[absIdx]
                gradients = []
            if absIdx == 0:
                chosenBucket = np.expand_dims(candidateBucket[actionVal], axis=0)
                chosenBucketLens = np.expand_dims(candidateBucketLens[actionVal], axis=0)
                chosenFeatBucket = np.expand_dims(candidateFeatBucket[actionVal], axis=0)
            else:
                chosenBucket = np.append(chosenBucket, [candidateBucket[actionVal]], axis=0)
                chosenBucketLens = np.append(chosenBucketLens, [candidateBucketLens[actionVal]], axis=0)
                chosenFeatBucket = np.append(chosenFeatBucket, [candidateFeatBucket[actionVal]], axis=0)

            candidateBucket = np.delete(candidateBucket, actionVal, 0)
            candidateBucketLens = np.delete(candidateBucketLens, actionVal, 0)
            candidateFeatBucket = np.delete(candidateFeatBucket, actionVal, 0)
            reward = self.single_reward_sketch(chosenBucket, chosenBucketLens)
            gradientBuff.append(gradients)
            rewardBuff.append(reward)
        return rewardBuff, gradientBuff

    def gen_sketch_sample(self):
        """Generating data sample."""
        if self.args.trainFlag:
            self.sampleClass = np.random.randint(self.numClasses)
            sampleIdx = np.random.randint(self.trainIterators[self.sampleClass].dataNum)
            self.sample = self.trainIterators[self.sampleClass].data_strokes[sampleIdx]
            self.sampleLens = self.trainIterators[self.sampleClass].len_strokes[sampleIdx]
            self.sampleFeat = self.trainIterators[self.sampleClass].data_feats[sampleIdx]
        else:
            self.sampleClass = self.testClass
            self.sample = self.testIterators[self.sampleClass].data_strokes[self.testIdx]
            self.sampleLens = self.testIterators[self.sampleClass].len_strokes[self.testIdx]
            self.sampleFeat = self.testIterators[self.sampleClass].data_feats[self.testIdx]
            self.testIdx += self.args.testIdxStep
            if self.testIdx >= self.testIterators[self.testClass].dataNum:
                self.testClass += 1
                self.testIdx = 0
                if self.args.testIdxStep == 1:
                    print("%d/%d" % (self.testClass, self.numClasses))
                    self.testing_output()
                if self.testClass >= self.numClasses:
                    self.args.trainFlag = True
                    self.agentRNNDropout, self.agentFCDropout = 0.5, 0.5
        zeroLenIdx = np.where(np.array(self.sampleLens) == 0)[0]
        if len(zeroLenIdx) > 0:
            self.sample = np.delete(self.sample, zeroLenIdx, axis=0)
            self.sampleLens = np.delete(self.sampleLens, zeroLenIdx, axis=0)
            self.sampleFeat = np.delete(self.sampleFeat, zeroLenIdx, axis=0)
        self.sampleBudget = min(self.classBudget[self.sampleClass], len(self.sample))
        self.sampleTimeStamp = [int(round((tempX / self.sample.shape[0]) * 10)) for tempX in range(self.sample.shape[0])]

    def single_reward_sketch(self, chosenBucket, chosenBucketLens):
        """Single step reward generation."""
        sketchArray = global_to_standard(chosenBucket.copy(), chosenBucketLens.copy())
        if self.args.classType == 'RNN':
            predProb, rankProb, rankIdx = ClassSketchRNN.model_pred(self.sess, self.classModel, np.expand_dims(sketchArray, axis=0), [len(sketchArray)])
        else:
            sketchRaster =  sketch_raster(np.append(np.array([[0, 0, 1]]), sketchArray.copy(), axis=0).copy(), preprocessImg=True, start_loc_val = self.startPos, normalizing_factor_val = self.normalizingFac)
            predProb, rankProb, rankIdx = ClassSketchCNN.model_pred(self.sess, self.classModel, sketchRaster)
        gtRank = np.where(rankIdx == self.sampleClass)[0][0]
        predClass = predProb.argmax()
        return predClass, predProb[predClass], np.where(rankIdx == self.sampleClass)[0][0], rankProb[gtRank]

    def gradientUpdate(self, gradBuff, rewardBuff):
        """Gradient update."""
        for rewardVal, gradVal in zip(rewardBuff, gradBuff):
            for gradIdx, gradItem in enumerate(gradVal):
                gradVal[gradIdx] = gradItem * rewardVal
        gradBuff = [sum(i) for i in zip(*gradBuff)]
        _ = self.sess.run(self.myAgent.update_grad, feed_dict=dict(zip(self.myAgent.gradient_holders, gradBuff)))

    def discount_rewards(self, rewardVals, gamma=0.99):
        """Discounted reward generation of the episode."""
        discountedReward = np.zeros_like(rewardVals)
        runningAdd = 0
        for t in reversed(range(0, rewardVals.size)):
            runningAdd = runningAdd * gamma + rewardVals[t]
            discountedReward[t] = runningAdd
        return discountedReward

    def reward_elaboration(self, rewardAgent, rewardHuman, rewardRandom, delta):
        """Reward elaboration of the episode."""
        rewardEp = np.zeros(len(rewardAgent), np.float32)
        for rewardIdx in range(len(rewardAgent)):
            rewardEp[rewardIdx] = (rewardAgent[rewardIdx][3] - (delta * rewardHuman[rewardIdx][3] + (1.0 - delta) * rewardRandom[rewardIdx][3])) * 100
        return list(self.discount_rewards(rewardEp, 0.90))

    def load_model(self, modelFile):
        """Loading abstraction agent weights."""
        modelWeights = np.load(modelFile).item()
        initOps = []
        modelVars = [var for var in tf.trainable_variables() if 'AbstractionAgent' in var.name]
        print('Loading Abstraction Model Weights')
        for var in modelVars:
            varName = var.name
            initOps.append(var.assign(modelWeights[varName]))
            print('var - %s' % (varName))
        print('Loading Abstraction Model Weights - Complete')
        return initOps

    def save_model(self):
        """Saving abstraction agent weights."""
        strModel = self.args.outDir + 'Agent_' + str(self.bestModelIdx) + '_Weights.npy'
        save_dict = {var.name: var.eval(session=self.sess) for var in tf.trainable_variables() if 'AbstractionAgent' in var.name}
        np.save(strModel, save_dict)
        self.bestModelAcc = self.testAcc
        self.bestModelIdx += 1
        if self.bestModelIdx == 5: self.bestModelIdx = 0

    def training_output(self, trainIdx):
        """Training step output."""
        print('Ep_idx: %d avgReward: %.2f avgAccAgent: %.2f avgAccUpperBound: %.2f avgAbsLen: %.2f avgFullLen: %.2f' % (trainIdx, np.mean(self.rewardList), np.mean(self.agentAccList) * 100, np.mean(self.upperAccList) * 100, np.mean(self.absLenList), np.mean(self.fullLenList)))

    def testing_output(self):
        """Testing step output."""
        print('Testing - avgAccAgent: %.2f avgAccUpperBound: %.2f avgAbsLen: %.2f avgFullLen: %.2f' % (np.mean(self.agentAccListTest) * 100, np.mean(self.upperAccListTest) * 100, np.mean(self.absLenListTest), np.mean(self.fullLenListTest)))

    def sketch_data_loading(self):
        """Loading sketch data."""
        self.classNames = ['cat', 'chair', 'face', 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe']
        self.classLens = [9.80, 4.85, 6.43, 8.28, 7.18, 9.08, 9.52, 3.57, 2.89]
        self.numClasses = len(self.classNames)
        self.classBudget = [math.ceil(classLen_ * self.args.budgetVal) for classLen_ in self.classLens]
        print('Loading Dataset')
        self.trainIterators, self.testIterators =  ([] for _ in range(2))
        for idx in range(self.numClasses):
            print('- %s' % (str(self.classNames[idx])))
            if self.args.trainFlag:
                iterator = np.load(self.args.dataDir + str(self.classNames[idx]) + '-train.npy').item()
                iterator.raw_data, iterator.seq_len = None, None
                iterator.dataNum = len(iterator.len_strokes)
                self.trainIterators.append(iterator)
            iterator = np.load(self.args.dataDir + str(self.classNames[idx]) + '-test.npy').item()
            iterator.raw_data, iterator.seq_len = None, None
            iterator.dataNum = len(iterator.len_strokes)
            self.testIterators.append(iterator)
        print('Loading Dataset Complete')

    def initialization(self):
        """Architecture initialization."""
        with tf.variable_scope('AbstractionAgent'): self.myAgent = Agent()
        self.classModel = ClassSketchRNN.model_initialization() if self.args.classType == 'RNN' else ClassSketchCNN.model_initialization()

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.args.gpuUsage)))
        self.sess.run(tf.global_variables_initializer())

        if self.args.agentLoading: self.sess.run(self.load_model(self.args.agentLoadingFile))
        ClassSketchRNN.model_loading(self.sess) if self.args.classType == 'RNN' else ClassSketchCNN.model_loading(self.sess)

        self.annealingFac = np.arange(0.0, 1.0, 1 / float(self.args.trainEps))
        self.bestModelAcc, self.bestModelIdx = -100.0, 0
        self.agentRNNDropout, self.agentFCDropout = 0.5, 0.5
        self.rewardList, self.agentAccList, self.upperAccList, self.absLenList, self.fullLenList = ([] for _ in range(5))

        if not os.path.exists(self.args.outDir): os.makedirs(self.args.outDir)
