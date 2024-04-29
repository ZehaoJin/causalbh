import numpy as np
import multiprocessing as mp
import os
import math
import uuid
import queue

from pathlib import Path
from tqdm.auto import tqdm, trange
from itertools import product, permutations


_trius, _num_variables, _root = None, None, None
def initializer(num_variables, root):
    global _trius, _num_variables, _root
    if _trius is None:
        _num_variables, _root = num_variables, root

        # Generate all the DAGs over num_variables nodes
        shape = (num_variables, num_variables)
        repeat = num_variables * (num_variables - 1) // 2

        # Generate all the possible binary codes
        codes = list(product([0, 1], repeat=repeat))
        codes = np.asarray(codes)

        # Get upper-triangular indices
        x, y = np.triu_indices(num_variables, k=1)

        # Fill the upper-triangular matrices
        _trius = np.zeros((len(codes),) + shape, dtype=np.int_)
        _trius[:, x, y] = codes


def generate_dags(inputs):
    global _trius, _num_variables, _root
    index, permutation = inputs

    # Permute the rows and columns
    permuted = _trius[:, :, permutation][:, permutation, :]
    permuted = permuted.reshape(-1, _num_variables ** 2)
    permuted = np.packbits(permuted, axis=1)

    # Save DAGs
    with open(_root / f'{index:04d}.npy', 'wb') as f:
        np.save(f, permuted)


def dispatcher(in_queue, out_queue):
    filenames = []
    while True:
        filename = in_queue.get()
        if filename is None:
            break
        filenames.append(filename)

        if len(filenames) >= 2:
            out_queue.put(tuple(filenames))
            filenames = []


def consumer(out_queue, in_queue, root, num_variables):
    while True:
        filenames = out_queue.get()
        if filenames is None:
            break

        dags_compressed = set()
        chunksize = math.ceil((num_variables ** 2) / 8)
        max_level = -1

        for filename, level in filenames:
            with open(root / filename, 'rb') as f:
                data = np.load(f)
                sequence = data.tobytes()
                for i in trange(0, len(sequence), chunksize):
                    dags_compressed.add(sequence[i:i + chunksize])

            max_level = max(max_level, level)

        dags_unique = np.frombuffer(b''.join(dags_compressed), dtype=np.uint8)
        dags_unique = dags_unique.reshape(-1, chunksize)

        filename = f'{uuid.uuid4()}.npy'
        with open(root / filename, 'wb') as f:
            np.save(f, dags_unique)

        for filename_, _ in filenames:
            (root / filename_).unlink()  # Delete file

        max_level = max_level + 1
        print(f'Create {filename} (level {max_level}) merging {filenames}')
        in_queue.put((filename, max_level))


if __name__ == '__main__':
    root = Path(os.getenv('SLURM_TMPDIR')) / 'dags'
    root.mkdir(exist_ok=True)

    num_variables = 7
    num_permutations = math.factorial(num_variables)

    with mp.Pool(60, initializer=initializer, initargs=(num_variables, root)) as pool:
        _ = list(tqdm(pool.imap_unordered(
            generate_dags, enumerate(permutations(range(num_variables)))),
        total=num_permutations))

    in_queue = mp.Queue()
    out_queue = mp.Queue()

    dispatch = mp.Process(target=dispatcher, args=(in_queue, out_queue), daemon=True)
    consumers = [
        mp.Process(target=consumer, args=(out_queue, in_queue, root, num_variables), daemon=True)
        for _ in range(60)
    ]

    for process in consumers:
        process.start()
    dispatch.start()

    for filename in root.iterdir():
        print(f'Adding {filename.relative_to(root)}')
        in_queue.put((filename.relative_to(root), 0))

    for process in consumers:
        process.join()
    dispatch.join()