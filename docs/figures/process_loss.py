import json
from pathlib import Path

with open(Path.cwd() / "loss.json", "r") as f:
    loss_dict = json.load(f)

length = min(len(loss_dict['frequency']), len(loss_dict['content']))
epoch = loss_dict['starting_epoch']

with open(Path.cwd() / "loss_content.txt", "w") as jc, open(Path.cwd() / "loss_frequency.txt", "w") as jf:
    jc.write("x y\n")
    jf.write("x y\n")
    for i in range(length):
        jc.write(f"{epoch} {loss_dict['content'][i]}\n")
        jf.write(f"{epoch} {loss_dict['frequency'][i]}\n")
        epoch += 1
