import numpy as np


def EvaluatePlanePair(plane_pair, eval_dict):
    a_res = []
    for ai in plane_pair[0]:
        #print(ai)
        #print(eval_dict)
        #print()
        a_res.append(ai.Evaluate(eval_dict))
    return (np.array(a_res), plane_pair[1].Evaluate(eval_dict))