import sys
import os

import argparse
import json

from time import time

sys.path.append(os.getcwd())

from src.timeseries.TimeSeriesLoader import uv_load

from  src.classification.WEASELClassifier import *
from  src.classification.BOSSEnsembleClassifier import *
from  src.classification.BOSSVSClassifier import *
from  src.classification.ShotgunEnsembleClassifier import *
from  src.classification.ShotgunClassifier import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name', type=str, default='solar_flare_2_regression')
    parser.add_argument('--base-path', type=str, default='d3m_datasets/eval_datasets/LL0')
    parser.add_argument('--seed', type=int, default=456)
    parser.add_argument('--use-schema', action='store_true')
    parser.add_argument('--no-print-results', action='store_true')
    parser.add_argument('--rparams', type=str, default='{}')
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)

prob_args = {
    "prob_name": args.prob_name,
    "base_path": args.base_path,
    "return_d3mds": True,
    "use_schema": args.use_schema,
    "strict": True,
}

t = time()

train, test = uv_load(args.base_path, args.prob_name)

import numpy as np
from sklearn.decomposition import PCA


def tss_to_numpy(bag=''):
    keys = list(bag.keys())

    remove = ['Type', 'Samples', 'Size', 'Labels']

    [keys.remove(r) for r in remove]

    return np.vstack([bag[i].data for i in keys])


def numpy_to_tss(bag='', m=''):
    keys = list(bag.keys())

    remove = ['Type', 'Samples', 'Size', 'Labels']
    [keys.remove(r) for r in remove]

    for i in keys:
        print(np.array(bag[i].data).sum())
        bag[i].data = m[i, :]
        print(np.array(bag[i].data).sum())
        print('-')

    return bag


def whiten(T_train='', T_test=''):
    pca = PCA(whiten=True).fit(np.vstack([T_train, T_test]))
    return pca.transform(T_train), pca.transform(T_test)


T_test, T_train = tss_to_numpy(test), tss_to_numpy(train)

T_train, T_test = whiten(T_train, T_test)

test = numpy_to_tss(test, T_test)
train = numpy_to_tss(train, T_train)





boss = BOSSEnsembleClassifier(args.prob_name)
scoreBOSS = boss.eval(train, test)

# weasel = WEASELClassifier(args.prob_name)
# scoreWeasel = weasel.eval(train, test)
# print(scoreWeasel)
res = {
    "prob_name": args.prob_name,
    "ll_metric": '',
    "ll_score": '',

    "test_score": scoreBOSS,

    "elapsed": time() - t,

    "_extra": {},
    "_misc": {
        "use_schema": args.use_schema,
    }
}
if not args.no_print_results:
    print(json.dumps(res))

# Save results
results_dir = os.path.join('results', os.path.basename(os.path.dirname(args.base_path)))
os.makedirs(results_dir, exist_ok=True)

result_path = os.path.join(results_dir, args.prob_name)
open(result_path, 'w').write(json.dumps(res) + '\n')
