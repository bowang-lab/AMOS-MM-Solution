import pandas as pd

from generate_green_score import GenerateGreenScore
from remove_normal import RemoveNormalFindings

path = "/home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/results/baseline/val_processed.csv"
df = pd.read_csv(path)

df.head()

RemoveNormalFindings(path, cache_dir="./CLEAN_model").run()
GenerateGreenScore(path,cache_dir="./GREEN_model").run()