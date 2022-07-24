from  minihit import minihit;
from src import barinel;
import utils;
import time;

PATHS = [
    '2172_ea4a3f8a',
    '3222_5729ed90',
    '4738_a7ce7f91',
    '4775_1ac05533',
    '4816_66bfc885',
    '4824_ad849602',
    '4877_6470c3f7',
    '4933_469d0096',
    '5043_2b1ce91d',
    '5250_6122df49',
    '5251_3d2393c7',
    '5251_6ce34ccf',
    '5259_a9e56e1e',
    '5398_19e7c1cd',
    '5442_a382917f',
    '5565_204849bc',
    '5565_44f4782a',
    '5655_96337372',
    '5687_3d2d8619',
    '5716_2d0ec942',
];


def diagnosis_Generator():
    for matrix_n, matrix_path in enumerate(PATHS):
        # load matrix from file 
        A, e, label, conflict = utils.build_matrix(f'matrices/{matrix_path}');
        rc_tree = minihit.RcTree(conflict)
        start_time: float = time.time();
        rc_tree.solve(prune=True, sort=False);
        D = list(map(lambda x: tuple(x),rc_tree.generate_minimal_hitting_sets()));
        output = barinel.barniel(A, e, D);
        runtime = time.time() - start_time;
        precision = utils.weighted_precision(output, set(label));
        recall = utils.weighted_recall(output, set(label));
        wasted = utils.wasted_effort(output, set(label));
        yield matrix_n, A.shape[1], A.shape[0], len(output),precision, recall, wasted , runtime;

if __name__ == '__main__':
    header = [
        'Martix no.',
        '# Components',
        '# Tests',
        '# Diagnoses',
        'Weighted Precision',
        'Weighted Recall',
        'Wasted Effort',
        'Runtime(ms)',  
    ];
    utils.write_to_csv('solutions.csv',header,diagnosis_Generator());