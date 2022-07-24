import numpy as np;
import json;

def build_matrix(path: str) -> tuple[np.ndarray, np.array,]:
    with open(path) as file:
        # get file info as json
        content: dict = json.load(file)
        conflict = list();
        test_n: int = len(content["initial_tests"]);
        mapper = {name: idx for idx, name in content['components_names']};
        comp_n: int = len(content['components_names']);
        bugs: list = [mapper[bug] for bug in content['bugs']];
        A: np.ndarray = np.zeros((test_n,comp_n));
        e: np.array = np.zeros(test_n);
        # fill matries
        for idx, (test_name, visited_comp, err) in enumerate(content['tests_details']):
            for c_idx in visited_comp:
                A[idx,c_idx] = 1;
            e[idx] = err;
            if err:
                conflict.append(set(visited_comp));
        return A,e,bugs,conflict;

def weighted_precision(predictions: list[set,float], label: set):
    precision: float = 0;
    sum: float = 0;
    for pred, score in predictions:
        correct = len(set(pred).intersection(label));
        precision += score * correct/ len(pred);
        sum += score;
    return precision / sum; # / sum - for floating point problem

def weighted_recall(predictions: list[set,float], label: set):
    recall: float = 0;
    sum: float = 0;
    for pred, score in predictions:
        correct = len(set(pred).intersection(label));
        recall += score * correct / len(label);
        sum += score;
    return recall / sum; # / sum - for floating point problem

def wasted_effort(predictions: list[set,float], label: set):
    need_to_fix: set = label.copy();
    wasted: set = set(); # components we already fixed
    for pred, _ in predictions:
        pred = set(pred);
        need_to_fix -= pred;
        wasted |= pred;
        if not need_to_fix:
            break;
    return len(wasted);