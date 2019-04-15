# -*- coding:utf-8 -*-
# Filename: AveragedPerceptron.py
# Author：hankcs
# Date: 2016-09-03 PM2:20
"""
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""
from collections import defaultdict
import pickle
import random


class AveragedPerceptron(object):

    '''An averaged perceptron, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    '''

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}  # 每个'位置'拥有一个权值向量
        self.classes = set()
        # 累加的权值，用于计算平均权值
        # 生成了一个默认为0的带key的数据字典
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)  # 上次更新权值时的i
        # 记录实例的数量
        self.i = 0

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)  # 生成一个默认dict,不存在的值返0.0
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight  # 每次预测一个特征值时该值都会重置
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda label: (scores[label], label))  # 返回得分最高的词性标签，如果得分相同取字母大的

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w  # 累加:(此时的i - 上次更新该权值时的i)*权值
            self._tstamps[param] = self.i  # 记录更新此权值时的i
            self.weights[f][c] = w + v  # 更新权值

        self.i += 1
        if truth == guess:
            return None
        for f in features:  # 遍历特征值,对每个特征值都加入当前判断正确和错误的词性,以及各自权值
            weights = self.weights.setdefault(f, {})  # 如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值,并将键值加入字典中,注意和get方法的区别
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None


    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged  # 向字典中加入key-value
            self.weights[feat] = new_feat_weights
        return None

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None


def train(nr_iter, examples):
    '''Return an averaged perceptron model trained on ``examples`` for
    ``nr_iter`` iterations.
    '''
    model = AveragedPerceptron()
    for i in range(nr_iter):
        random.shuffle(examples)
        for features, class_ in examples:
            scores = model.predict(features)
            guess, score = max(scores.items(), key=lambda i: i[1])
            if guess != class_:
                model.update(class_, guess, features)
    model.average_weights()
    return model