from scipy.optimize import minimize;
import numpy as np;

def probability_function(A: np.ndarray, e: np.array, D: list[set], P: float):
    """
    """
    def maxmize(x):
        """
        """
        math = np.multiply(partial_A, x);
        probability = list();
        for test, obs in enumerate(partial_e):
            partic_idx = np.where(math[test] != 0)[0];
            prod = np.prod(math[test][partic_idx]);
            probability.append(obs * (1-prod) + (1-obs)*prod)
        return -np.prod(probability);
    probabilities: list = []
    for diagnosis in D:
        # init state to search from
        guess: np.array = np.ones(len(diagnosis)) * P;
        # select all information about components in matrix
        partial_A: np.ndarray = A[:, diagnosis];
        # get all the rows witch include atleast one of the components
        rows_inclusion: np.array = np.where(np.sum(partial_A, axis=1) != 0)[0];
        partial_A = partial_A[rows_inclusion, :];
        # get rows of error
        partial_e = e[rows_inclusion];
        # boudouries for scipy for each component (probabilities is between 0 and 1);
        bounds = [(0,1)] * len(diagnosis);
        #  Use Gradient optimizer to solve the problem ( newthon solver for more accurate sulation);
        probabilities.append(-minimize(maxmize, guess, method='L-BFGS-B', bounds=bounds).fun);
    return probabilities;

def pass_although_broken(diagnosis: set, m: int, P: float):
    return np.power(P,len(diagnosis)) * np.power(1-P, m-len(diagnosis));

def barniel(A: np.ndarray, e: np.array, D: list[set], P: float=0.01):
    m: int = A.shape[1];
    # probability function after optimization
    probabilities: np.array = probability_function(A, e, D, P);
    # get posability of observation (without past probabilities data)
    obs_prob = np.array([pass_although_broken(diagnosis, m,P) for diagnosis in D]);
    # get the multaplication probability;
    mul = probabilities * obs_prob;
    # normalize_it , we didn't use the normalization of pr(obs) because it dont change the ranked
    probabilities = mul / np.sum(mul);
    return sorted(
        list(zip(D,probabilities)),
        key= lambda item: item[1],
        reverse=True,
    );