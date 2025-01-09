import numpy as np
import time

n = 7
MEC = np.load(f'/data/zj448/causal/exact_posteriors/MEC_{n}.npy')
#MEC = MEC[:10000, :, :]

start_time = time.time()
unique_MECs, index, counts = np.unique(MEC, axis=0, return_counts=True, return_index=True)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")


np.save(f'/data/zj448/causal/exact_posteriors/unique_MECs_{n}.npy', unique_MECs)
np.save(f'/data/zj448/causal/exact_posteriors/unique_MEC_index_{n}.npy', index)
np.save(f'/data/zj448/causal/exact_posteriors/unique_MEC_counts_{n}.npy', counts)



