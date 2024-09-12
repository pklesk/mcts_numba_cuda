import matplotlib.pyplot as plt
import numpy as np

def data_plot(data, labels_x, labels_y, title_1, title_2, details=[], details_labels=[]):
    
    figsize = (6, 6)
    fontsize_suptitle = 14
    fontsize_title = 14
    fontsize_main = 17
    fontsize_details = 10
    
    plt.figure(figsize=figsize)    
    if title_1:
        plt.suptitle(title_1, fontsize=fontsize_suptitle)
    if title_2:
        plt.title(title_2, fontsize=fontsize_title)     
    
    plt.imshow(data, cmap="coolwarm", origin="lower", vmin=0.0, vmax=1.0)

    plt.xticks(ticks=[0, 1, 2, 3], labels=labels_x)
    plt.yticks(ticks=[0, 1, 2, 3], labels=labels_y)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j] * 100:.0f}%", ha="center", va="center", color="black", fontsize=fontsize_main)
            for k in range(len(details)):
                plt.text(j, i - 0.1 - 0.1 * k, f"{details_labels[k]}: {details[k][i, j]:.1f}", ha='center', va='center', color='black', fontsize=fontsize_details)
     
    plt.tight_layout(pad=0.4) 
    plt.show()

if __name__ == "__main__":
    print("plots main")
    data = np.random.rand(4, 4) * 0.5 + 0.5
    details = [np.random.rand(4, 4) * 1000.0,
               np.random.rand(4, 4) * 10.0,
               10.0 + np.random.rand(4, 4) * 10.0]
    details = []
    details_labels = ["steps", "depths", "max depths"]
    labels_x = [32, 64, 128, 256]
    labels_y = [1, 2, 4, 8]
    title_1 = ""#SCORES (CONNECT 4)"
    title_2 = "mctsnc_1_inf_8_64_ocp_thrifty vs mcts_4_inf_vanilla".upper()
    data_plot(data, labels_x, labels_y, title_1, title_2, details, details_labels)
