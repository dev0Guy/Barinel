"""
    This Implements Scatto Paper (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.169.1046&rep=rep1&type=pdf)
"""

#############################
##        Import            #              
#############################
from audioop import reverse
from collections import namedtuple
import numpy as np;

__all__ = ['staccato'];

OchRanking = namedtuple("OchRanking", ["score","n11"]);

def _ranked(A: np.ndarray, e: np.array, comp_idx: int):
    spectra = A[:, comp_idx].reshape(1,-1);
    n11: float = np.sum(spectra * e);
    n10: float = np.sum(spectra * (1 - e));
    n01: float = np.sum((1 -spectra) * e);
    mult: float = (n11 + n10) * (n11 + n01);
    score: float = 0;
    if mult != 0:
       score = n11 / np.sqrt(mult);
    return OchRanking(score, n11);

def _strip_components(A: np.ndarray, indexes: list):
    return np.copy(A[:, indexes]);

def _update_components_map(componets: dict, rmv_comp: list):
    sub_array = np.zeros(len(componets),dtype=np.int);
    for _from in sorted(rmv_comp):
        _from = int(list(_from)[0]);
        sub_array[_from] = -_from;
        sub_array[_from+1:] -= 1;
    for comp_idx, _ in enumerate(componets):
        componets[comp_idx] += sub_array[comp_idx];

def _strip(A: np.ndarray, e: np.array, components: list, comp: int):
    conflict_not_involved = np.where(A[:, components[comp]] == 0)[0];
    others = [components[idx] for  idx in components if idx != comp];
    return np.copy(A[conflict_not_involved,:][:,others]), np.copy(e[conflict_not_involved]);

def _is_superset(component: set, diag: list[set]):
    return any(component.issuperset(elem) for elem in diag);

def _remove_superset(diag: list[set]):
    for idx, val in enumerate(diag):
        if _is_superset(set(val), diag[:idx]+ diag[idx+1:]):
            diag.remove(val);

def staccato(A: np.ndarray, e: np.array, components: list[int]=None,comp_n:int=None, lam: float = 0.8, l: int=25):
    """
        @Todo:  Implement
    """
    # create local copy of matrixes
    A, e =  np.copy(A), np.copy(e);
    comp_n: int = A.shape[1] if not comp_n else comp_n;
    components: list[int] = [idx for idx in range(comp_n)] if not components else components.copy();
    diagnosis: list[set] = [];
    conflict_num: int = (int)(np.sum(e));
    seen: float = 0;
    ranked: dict = {comp_idx: _ranked(A,e,comp_idx) for comp_idx in range(comp_n)};
    ranked_mapper: list = list(dict(sorted(ranked.items(), key=lambda item: item[1].score, reverse=True)).keys());
    for comp_idx in range(comp_n):
        if ranked[comp_idx].n11 == conflict_num:
            diagnosis.append({comp_idx});
            ranked_mapper.remove(comp_idx);
            seen += 1 / comp_n;
    A = _strip_components(A, ranked_mapper);
    _update_components_map(components, diagnosis);
    while len(ranked_mapper) > 0 and seen <= lam and len(diagnosis) < l:
        components_cpy = components.copy();
        seen += 1 / comp_n;
        unit_component = {ranked_mapper.pop(0)};
        A_hat, e_hat = _strip(A, e, components_cpy, list(unit_component)[0]);
        _update_components_map(components_cpy, [unit_component]);
        diagnosos_hat = staccato(A_hat, e_hat,components=components_cpy,comp_n=None, lam=lam-seen, l=l-len(diagnosis));
        while diagnosos_hat and len(diagnosos_hat) > 0:
            comp = diagnosos_hat.pop(0);
            comp |= unit_component;
            if not _is_superset(comp, diagnosis):
                diagnosis.append(comp);
    _remove_superset(diagnosis);
    return diagnosis;
