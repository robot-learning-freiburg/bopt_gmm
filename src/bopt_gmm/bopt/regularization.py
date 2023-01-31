import numpy as np
import sys

from dataclasses import dataclass
from typing      import Callable, Any

def f_zero(model_new, model_base, points):
    return 0


def f_joint_prob(model_new, model_base, points):
    pos   = points[:, model_base.state_dim]
    base_inf = model_base.predict(pos, model_base.state_dim, full=True)
    new_inf  = model_new.predict(pos, model_base.state_dim, full=True)
    try:
        cond_prob_base, cond_prob_new = model_base.conditional_pdf(pos, model_base.state_dim, base_inf, new_inf)
    except ValueError as e:
        return 0

    # joint_prob = np.log(cond_prob_new) / np.log(cond_prob_base)
    # conditional  = np.nan_to_num(joint_prob / pos_prob)
    # conditional /= conditional.max() + sys.float_info.epsilon
    return np.log(cond_prob_new).sum() / np.log(cond_prob_base).sum()


def f_mean_prob(model_new, model_base, points):
    pos   = points[:, model_base.state_dim]
    base_inf = model_base.predict(pos, model_base.state_dim, full=True)
    new_inf  = model_new.predict(pos, model_base.state_dim, full=True)
    try:
        cond_prob_base, cond_prob_new = model_base.conditional_pdf(pos, model_base.state_dim, base_inf, new_inf)
    except ValueError as e:
        return 0

    return cond_prob_new.mean() / cond_prob_base.mean()


def f_kl(model_new, model_base, points):
    p_mb = model_base.pdf(points, tuple(range(points.shape[1])))
    p_mn = model_new.pdf(points, tuple(range(points.shape[1])))
    return np.sum(np.where(p_mb != 0, p_mb * np.log(p_mb / p_mn), 0))


def f_jsd(model_new, model_base, points):
    p_mb = model_base.pdf(points, tuple(range(points.shape[1])))
    p_mn = model_new.pdf(points, tuple(range(points.shape[1])))
    return 0.5 * (np.sum(np.where(p_mb != 0, p_mb * np.log(p_mb / p_mn), 0)) + 
                  np.sum(np.where(p_mn != 0, p_mn * np.log(p_mn / p_mb), 0)))


def f_dot(model_new, model_base, points):
    i = tuple(range(points.shape[1]))
    inv_mb = model_base.predict(points, i, full=True)
    b_norm = np.sqrt((inv_mb ** 2).sum(axis=1))
    inv_mb = (inv_mb.T / b_norm).T
    inv_mn = (model_new.predict(points, i, full=True).T / b_norm).T

    return np.nan_to_num((inv_mb * inv_mn).sum(axis=1), nan=1.0).mean()


def gen_regularizer(cfg):
    if cfg is None or cfg.f == 'zero':
        return f_zero
    
    if cfg.f == 'p_joint':
        def f_reg(model_new, model_base, points):
            return f_joint_prob(model_new, model_base, points) * cfg.b
    elif cfg.f == 'p_mean':
        def f_reg(model_new, model_base, points):
            return f_mean_prob(model_new, model_base, points) * cfg.b
    elif cfg.f == 'kl':
        def f_reg(model_new, model_base, points):
            return f_kl(model_new, model_base, points) * cfg.b
    elif cfg.f == 'jsd':
        def f_reg(model_new, model_base, points):
            return f_jsd(model_new, model_base, points) * cfg.b
    elif cfg.f == 'dot':
        def f_reg(model_new, model_base, points):
            return np.exp(-np.abs(f_kl(model_new, model_base, points))) * cfg.b

    return f_reg

