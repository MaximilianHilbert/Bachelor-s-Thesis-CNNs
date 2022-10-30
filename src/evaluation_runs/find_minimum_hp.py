import os
import numpy as np
from tqdm import tqdm
full_lst=os.listdir("evaluation_errors3l")
curr_value=float(10)
name=""
for file_idx in tqdm(range(len(full_lst))):
    filename=full_lst[file_idx]
    with open("evaluation_errors3l/"+filename, "r") as f:
        lines=f.readlines()
        median=np.median(np.array(lines, dtype=np.float32))
    if median<curr_value:
        curr_value=median
        name=filename
print(curr_value, name)