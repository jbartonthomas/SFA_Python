import sys
import os

import argparse

sys.path.append(os.getcwd())

from src.timeseries.TimeSeriesLoader import uv_load

from  src.classification.WEASELClassifier import *
from  src.classification.BOSSEnsembleClassifier import *
from  src.classification.BOSSVSClassifier import *
from  src.classification.ShotgunEnsembleClassifier import *
from  src.classification.ShotgunClassifier import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name',        type=str, default='solar_flare_2_regression')
    parser.add_argument('--base-path',        type=str, default='d3m_datasets/eval_datasets/LL0')
    parser.add_argument('--seed',             type=int, default=456)
    parser.add_argument('--use-schema',       action='store_true')
    parser.add_argument('--no-print-results', action='store_true')
    parser.add_argument('--rparams',          type=str, default='{}')
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)



prob_args = {
    "prob_name"    : args.prob_name,
    "base_path"    : args.base_path,
    "return_d3mds" : True,
    "use_schema"   : args.use_schema,
    "strict"       : True,
}



train, test = uv_load(args.base_path, args.prob_name)

boss = BOSSEnsembleClassifier(args.prob_name)
scoreBOSS = boss.eval(train, test)[0]
# print(data+"; "+scoreBOSS)

res = {
    "prob_name" : args.prob_name,
    "ll_metric" : '',
    "ll_score"  : '', 
    
    "test_score" : scoreBOSS,
    
    "elapsed" : time() - t,
    
    "_extra" : _extra,
    "_misc"  : {
        "use_schema" : args.use_schema,
    }
}
if not args.no_print_results:
    print(json.dumps(res))

# Save results
results_dir = os.path.join('results', os.path.basename(os.path.dirname(args.base_path)))
os.makedirs(results_dir, exist_ok=True)

result_path = os.path.join(results_dir, args.prob_name)
open(result_path, 'w').write(json.dumps(res) + '\n')
