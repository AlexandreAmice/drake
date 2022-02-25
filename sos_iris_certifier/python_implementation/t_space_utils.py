import numpy as np


def convert_t_to_q(t, t_star = None):
    if t_star is not None:
        t = t-t_star
    q = np.arctan2(2*t/(1+t**2), (1-t**2)/(1+t**2))
    return q #q[:,[0,2,1]]

def convert_q_to_t(q, q_star = None):
    if q_star is not None:
        q = q-q_star
    t = np.tan(q/2)
    return t#np.tan(np.divide(q,2))[:,[0,2,1]]

def EvaluatePlanePair(plane_pair, eval_dict):
    a_res = []
    for ai in plane_pair[0]:
        a_res.append(ai.Evaluate(eval_dict))
    return (np.array(a_res), plane_pair[1].Evaluate(eval_dict))

