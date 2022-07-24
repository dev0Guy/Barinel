from src import barinel, staccato;
import numpy as np;
import json;

def build_matrix(path: str) -> tuple[np.ndarray, np.array]:
    with open(path) as file:
        # get file info as json
        content: dict = json.load(file)
        test_n: int = len(content["initial_tests"]);
        comp_n: int = len(content['components_names']);
        A: np.ndarray = np.zeros((test_n,comp_n));
        e: np.array = np.zeros(test_n);
        # fill matries
        for idx, (test_name, visited_comp, err) in enumerate(content['tests_details']):
            for c_idx in visited_comp:
                A[idx,c_idx] = 1;
            e[idx] = err;
        return A,e;

# load matrix from file 
A, e = build_matrix('matrices/2172_ea4a3f8a');
# call stccato
D = tuple(staccato(A,e));
D = [tuple(x) for x in D];
# run 
output = barinel.barniel(A, e, D);
print(output)