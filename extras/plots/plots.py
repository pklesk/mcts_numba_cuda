import matplotlib.pyplot as plt
import numpy as np
from utils import unzip_and_load_experiment

FOLDER_EXPERIMENTS = "../../experiments/"

def scores_plot(data, details, label_x, label_y, ticks_x, ticks_y, title_1, title_2):    
    figsize = (6, 6)
    fontsize_suptitle = 20
    fontsize_title = 20
    fontsize_ticks = 14
    fontsize_labels = 19
    fontsize_main = 20
    fontsize_details = 13       
    plt.figure(figsize=figsize)    
    if title_1:
        plt.suptitle(title_1, fontsize=fontsize_suptitle)
    if title_2:
        plt.title(title_2, fontsize=fontsize_title)         
    plt.imshow(data, cmap="coolwarm", origin="lower", vmin=0.0, vmax=1.0)    
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.xticks(ticks=np.arange(data.shape[1]), labels=ticks_x, fontsize=fontsize_ticks)
    plt.yticks(ticks=np.arange(data.shape[0]), labels=ticks_y, fontsize=fontsize_ticks)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j] * 100:.1f}%", ha="center", va="center", color="black", fontsize=fontsize_main)
            if len(details) > 0:
                for k in range(len(details)):
                    plt.text(j, i - 0.25 - 0.15 * k, details[k][i, j], ha='center', va='center', color='black', fontsize=fontsize_details)     
    plt.tight_layout(pad=0.4) 
    plt.show()

def scores_plot_generator(experiments_hs_array, label_x, label_y, ticks_x, ticks_y, title_1, title_2):
    print("SCORES PLOT GENERATOR...")    
    data = np.zeros(experiments_hs_array.shape)
    details_steps = np.empty(experiments_hs_array.shape, dtype=object)
    details_depths = np.empty(experiments_hs_array.shape, dtype=object)
    for i in range(experiments_hs_array.shape[0]):
        for j in range(experiments_hs_array.shape[1]):
            experiment_info = unzip_and_load_experiment(experiments_hs_array[i, j], FOLDER_EXPERIMENTS)
            data[i, j] = experiment_info["stats"]["score_b_mean"]
            n_games = experiment_info["matchup_info"]["n_games"]
            steps = []
            mean_depths = []
            max_depths = []            
            for g in range(n_games):
                moves_rounds = experiment_info["games_infos"][str(g + 1)]["moves_rounds"]
                bw_prefix = "white_" if g % 2 == 0 else "black_"
                for m in range(len(moves_rounds)):
                    moves_round = moves_rounds[str(m + 1)]
                    bwpi = bw_prefix + "performance_info"
                    if bwpi in moves_round:
                        steps.append(moves_round[bwpi]["steps"])
                        mean_depths.append(moves_round[bwpi]["trees"]["mean_depth"])
                        max_depths.append(moves_round[bwpi]["trees"]["max_depth"])
            details_steps[i, j] = f"{np.mean(steps):.1f}"
            details_depths[i, j] = f"{np.mean(mean_depths):.1f}/{np.mean(max_depths):.1f}"
    details = [details_steps, details_depths]        
    scores_plot(data, details, label_x, label_y, ticks_x, ticks_y, title_1, title_2)
    print("SCORES PLOT GENERATOR DONE.")

def scores_plot_ocp_thrifty_vs_vanilla():
    experiments_hs_array = np.array([
        ["0295905193_59892_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_32_ocp_thrifty;C4_6x7;100]",
         "3448791497_42324_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_64_ocp_thrifty;C4_6x7;100]",
         "0885591745_06572_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_128_ocp_thrifty;C4_6x7;100]",
         "0333431545_83812_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_256_ocp_thrifty;C4_6x7;100]"],
        ["2759297865_05556_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_32_ocp_thrifty;C4_6x7;100]",
         "1617216873_20692_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_64_ocp_thrifty;C4_6x7;100]",
         "0864518467_53902_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_128_ocp_thrifty;C4_6x7;100]",
         "0312358267_31142_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_256_ocp_thrifty;C4_6x7;100]"],
        ["3391115913_62292_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_32_ocp_thrifty;C4_6x7;100]",
         "2249034921_44724_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_ocp_thrifty;C4_6x7;100]",
         "0822371911_48562_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_128_ocp_thrifty;C4_6x7;100]",
         "0270211711_58506_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_256_ocp_thrifty;C4_6x7;100]"],
        ["0359784713_10356_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_32_ocp_thrifty;C4_6x7;100]",
         "3512671017_25492_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_64_ocp_thrifty;C4_6x7;100]",
         "0738078799_70586_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_128_ocp_thrifty;C4_6x7;100]",
         "0185918599_47826_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_256_ocp_thrifty;C4_6x7;100]"]
        ])
    scores_plot_generator(experiments_hs_array, "n_playouts (m)", "n_trees (T)", [32, 64, 128, 256], [1, 2, 4, 8], None, "OCP-THRIFTY 1s (vs VANILLA MCTS 4s)")    

if __name__ == "__main__":        
    scores_plot_ocp_thrifty_vs_vanilla()