import numpy as np
import time

MEC = np.load('/data/zj448/causal/exact_posteriors/MEC_7.npy')
#MEC = MEC[:10000, :, :]

start_time = time.time()
unique_MECs, index, counts = np.unique(MEC, axis=0, return_counts=True, return_index=True)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")


np.save('/data/zj448/causal/exact_posteriors/unique_MECs_7.npy', unique_MECs)
np.save('/data/zj448/causal/exact_posteriors/unique_MEC_index_7.npy', index)
np.save('/data/zj448/causal/exact_posteriors/unique_MEC_counts_7.npy', counts)



