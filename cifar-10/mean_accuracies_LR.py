#%%
import pandas as pd
import os

#%%
df = pd.concat(
    pd.read_csv(f_name).assign(fname=f_name)
    for f_name in os.listdir()
    if f_name.endswith("csv") and not "trash" in f_name
)
df["model"], df["activation"], df["lrate"], _ = (
    df["fname"].str.split("_", expand=True).values.T
)

# %%
df = (
    df[df["Epoch"] == 99]
    .drop(columns=["Unnamed: 0", "fname"])
    .groupby(["model", "activation", "lrate"])
    .mean()
    .reset_index()
)

# %%
df["lrate"] = df.lrate.str.strip("LR").astype(float)

# %%
for model in df.model.unique():
    tmp = df[df["model"] == model].pivot_table(
        index="lrate", columns="activation", values=["Test accuracy"]
    )
    tmp.columns = tmp.columns.get_level_values(1)
    tmp.reset_index().rename(columns={"lrate": "LR"}).to_latex(
        f"{model}_testAccuracy.tex", column_format=f"c||ccccc", index=False
    )

for model in df.model.unique():
    tmp = df[df["model"] == model].pivot_table(
        index="lrate", columns="activation", values=["Test top-3 accuracy"]
    )
    tmp.columns = tmp.columns.get_level_values(1)
    tmp.reset_index().rename(columns={"lrate": "LR"}).to_latex(
        f"{model}_testTop3Accuracy.tex", column_format=f"c||ccccc", index=False
    )

# %%
for f in os.listdir():
    if not f.endswith(".tex"):
        continue
    prefix = [
        r"\documentclass[margin=0.5cm]{standalone}",
        "\n" r"\usepackage{booktabs}",
        "\n",
        r"\begin{document}",
        "\n",
    ]
    suffix = [r"\end{document}"]
    with open(f) as fp:
        lines = fp.readlines()
    lines = prefix + lines + suffix
    with open(f, "w") as fp:
        fp.writelines(lines)

# %%
