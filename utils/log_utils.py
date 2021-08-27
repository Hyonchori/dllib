
import csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_name = "../weights/keypoint_detector_log.csv"
    with open(file_name) as f:
        reader = csv.reader(f)
        logs = np.array(list(reader)).astype(np.float32)

        plt.subplot(3, 1, 1)
        plt.plot(logs[..., 0])
        plt.plot(logs[..., 3])

        plt.subplot(3, 1, 2)
        plt.plot(logs[..., 1])
        plt.plot(logs[..., 4])

        plt.subplot(3, 1, 3)
        plt.plot(logs[..., 2])
        plt.plot(logs[..., 5])

        plt.show()
