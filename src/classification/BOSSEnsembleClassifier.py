from src.transformation.BOSS import *

from joblib import Parallel, delayed

from exline.modeling.metrics import metrics
from time import time
import pandas as pd

from multiprocessing import Pool

import pyximport;

pyximport.install()

from src.classification import cBOSSEnsembleClassifier

'''
The Bag-of-SFA-Symbols Ensemble Classifier as published in
 Schäfer, P.: The boss is concerned with time series classification
 in the presence of noise. DMKD (2015)
'''


def _predict(args):
    self, samples, model = args
    wordsTest = model[3].createWords(samples)
    test_bag = model[3].createBagOfPattern(wordsTest, samples, model[2])

    p_correct, p_labels = self.prediction(test_bag, model[5], samples["Labels"], model[6], True)

    return p_labels


class BOSSEnsembleClassifier():
    def __init__(self, d, test=False):
        self.NAME = d
        self.factor = 0.96
        self.maxF = 16
        self.minF = 6
        self.maxS = 4
        self.MAX_WINDOW_LENGTH = 250

        if test:
            self.MAX_WINDOW_LENGTH = 25

    def eval(self, train, test):
        print('model fit')
        scores = self.fit(train)

        # parallel
        labels, correctTesting = self.predict(self.model, test)

        test_acc = correctTesting / test["Samples"]

        f1 = metrics['f1Macro'](labels, test['Labels'])

        return {'f1Macro': f1, 'accuracy': test_acc}

    def fit(self, train):
        self.minWindowLength = 10
        maxWindowLength = self.MAX_WINDOW_LENGTH
        for i in range(train["Samples"]):
            maxWindowLength = min([len(train[i].data), maxWindowLength])

        self.windows = range(maxWindowLength, self.minWindowLength, -1)

        NORMALIZATION = [True, False]
        bestCorrectTraining = 0.
        bestScore = None

        for norm in NORMALIZATION:
            models, correctTraining = self.fitEnsemble(norm, train)
            labels, correctTesting = self.predict(models, train)

            if bestCorrectTraining < correctTesting:
                bestCorrectTraining = correctTesting
                bestScore = correctTesting / train["Samples"]
                self.model = models

        return bestScore

    def fitIndividual(self, args):
        NormMean, samples, i= args
        model = self.BOSSModel(NormMean, self.windows[i])

        boss = BOSS(self.maxF, self.maxS, self.windows[i], NormMean)
        train_words = boss.createWords(samples) # can't parallelize createWords for the predict step b/c nested here

        f = self.minF
        keep_going = True
        while (f <= self.maxF) & (keep_going == True):
            bag = boss.createBagOfPattern(train_words, samples, f)
            s = self.prediction(bag, bag, samples["Labels"], samples["Labels"], False)
            if s[0] > model[1]:
                model[1] = s[0]
                model[2] = f
                model[3] = boss
                model[5] = bag
                model[6] = samples["Labels"]
            if s[0] == samples["Samples"]:
                keep_going = False
            f += 2

        # self.results.append(model)
        #bar.update(i)
        return model

    def fitEnsemble(self, NormMean, samples):
        correctTraining = 0
        self.results = []
        from multiprocessing import Pool
        print(self.NAME + "  Fitting for a norm of " + str(NormMean))

        import tqdm

        args = [(NormMean, samples, i) for i in range(len(self.windows))]
        results = []
        pool = Pool(processes=32)
        for _ in tqdm.tqdm(pool.imap_unordered(self.fitIndividual, args), total=len(args)):
            results.append(_)

        print()
        self.results = results

        # Find best correctTraining
        for i in range(len(self.results)):
            if self.results[i][1] > correctTraining:
                correctTraining = self.results[i][1]

        # Remove Results that are no longer satisfactory
        new_results = []
        for i in range(len(self.results)):
            if self.results[i][1] >= (correctTraining * self.factor):
                new_results.append(self.results[i])

        return new_results, correctTraining

    def BossScore(self, normed, windowLength):
        return ["BOSS Ensemble", 0, 0, normed, windowLength, pd.DataFrame(), 0]

    def BOSSModel(self, normed, windowLength):
        return self.BossScore(normed, windowLength)

    def prediction(self, bag_test, bag_train, label_test, label_train, training_check):
        p_correct, p_labels = cBOSSEnsembleClassifier.prediction(bag_test, bag_train, label_test, label_train,
                                                                 training_check)

        return p_correct, p_labels



    def predict(self, models, samples):

        Label_Matrix = np.zeros((samples["Samples"], len(models)))
        Label_Vector = np.zeros((samples["Samples"]))

        t0 = time()

        p = Pool(32)
        labels = p.map(_predict, [(self, samples, m) for m in models])

        print('first loop took: {}s'.format(time() - t0))

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                Label_Matrix[j, i] = labels[i][j]

        # samples x models matrix

        print('first loop took: {}s'.format(time() - t0))

        unique_labels = np.unique(samples["Labels"])
        t0 = time()
        for i in range(len(Label_Vector)):
            maximum = 0
            best = 0
            unique, counts = np.unique(Label_Matrix[i, :], return_counts=True)
            d = dict(zip(unique, counts))

            for key in unique_labels:
                key = float(key)
                try:
                    if d[key] > maximum:
                        maximum = d[key]
                        best = key
                except KeyError:
                    pass

            Label_Vector[i] = best
        print('second loop took: {}s'.format(time() - t0))

        t0 = time()
        correctTesting = 0
        for i in range(len(Label_Vector)):
            if int(Label_Vector[i]) == int(samples["Labels"][i]):
                correctTesting += 1
        print('third loop took: {}s'.format(time() - t0))

        return Label_Vector, correctTesting