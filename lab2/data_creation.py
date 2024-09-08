# data_creation.py
#!/usr/bin/python3
import os
import numpy as np

def create_data(data_dir, num_samples=1000):
    os.makedirs(data_dir, exist_ok=True)

    for i in range(num_samples):
        x = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 0.1, 100)
        y = 2 * x + 1 + noise

        if i < num_samples // 2:
            np.save(os.path.join(data_dir, "train_data_{}.npy".format(i)), (x, y))
        else:
            np.save(os.path.join(data_dir, "test_data_{}.npy".format(i)), (x, y))

if __name__ == "__main__":
    create_data("train")
    create_data("test")