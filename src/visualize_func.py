#!/usr/bin/env python3
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

sns.set()

def visualize_func():
    with open("outputs.pkl", "rb") as fin:
        outputs = pickle.load(fin)

    with open("labels.pkl", "rb") as fin:
        labels = pickle.load(fin)

    with open("loss.pkl", "rb") as fin:
        loss = pickle.load(fin)


    outputs = np.array(outputs)
    outputs = np.squeeze(outputs)

    labels = np.array(labels)
    labels = np.squeeze(labels)

    vmin = min(outputs.min(), labels.min())
    vmax = max(outputs.max(), labels.max())

    r2 = pearsonr(labels, outputs).statistic
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(outputs, labels, alpha=0.7)
    plt.xlim([vmin, vmax])
    plt.ylim([vmin, vmax])

    ax.set_xlabel("Acutal valus", fontsize=15)
    ax.set_ylabel("Predictions", fontsize=15)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="red")
    ax.set_title(f"$R^2$: {r2:.3f}, MSE Loss: {loss:.3f}", fontsize=20)
    
    plt.show()

if __name__ == "__main__":
    visualize_func()

    
    
    
