import csv
import numpy as np
from matplotlib import pyplot as plt

filename = "ADDO2012A036.WAV_a0104_71.f0.csv"

with open("/Users/dani/Documents/GitHub/TF-Animalysis/data/test_rumbles/f0_pesto/"+filename, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip the header
    data = np.array(list(reader)).astype(float)

with open("/Users/dani/Documents/GitHub/TF-Animalysis/data/test_rumbles/f0_refined/"+filename, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip the header
    data2 = np.array(list(reader)).astype(float)

data[:, 0] = data[:, 0] * 1000  # convert time to milliseconds
# resample from 10 ms to 16 ms with scipy   
from scipy.signal import resample
new_length = int(len(data) * 10 / 16)
data_resampled = resample(data, new_length)

print(data_resampled.shape)

plt.figure(figsize=(10, 4))
plt.plot(data_resampled[:, 1], label="F0 Contour")
plt.plot(data2[:, 1]*4, label="Refined F0 Contour")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("F0 Contour from CSV")
plt.show()