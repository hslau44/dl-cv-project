import numpy as np


def generate_int(r):
    if len(r) != 2:
        raise ValueError()
    v = np.random.randint(r[0],r[1])
    return v

GEN_DIC = {
    'int': generate_int
}


def get_testdata(num_samples, types, ranges):
    """
    
    ------ Example ------
    get_testdata(
        num_sample=6,
        types=[
            'int',
        ], 
        ranges=[
            [1,4],
            [2,16],
        ],
    )
    """
    data = []
    for i in range(num_samples):
        row = []
        for t, r in zip(types, ranges):
            value = GEN_DIC[t](r)
            row.append(value)
        data.append(row)
    return data