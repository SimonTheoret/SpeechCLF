"quick fix"

import numpy as np
import pandas as pd

data = np.random.rand(10, 4)
df = pd.DataFrame(data, columns=["val1", "val2", "tweet", "labels"])
df["tweet"] = "some tweet"
df["labels"] = 1

df.to_csv("data/fake_dataset.csv")
