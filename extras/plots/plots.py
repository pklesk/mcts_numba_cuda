import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import numpy as np
from utils import unzip_and_load_experiment

FOLDER_EXPERIMENTS = "../../experiments/"

def scores_plot(data, details, label_x, label_y, ticks_x, ticks_y, title_1, title_2):    
    figsize = (6, 6)
    fontsize_suptitle = 20
    fontsize_title = 23
    fontsize_ticks = 16
    fontsize_labels = 20
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
    scores_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], None, "MCTS-NC OCP-THRIFTY (1s) \nvs MCTS VANILLA (4s)")    

def best_action_plot(steps_black, qs_black, ucbs_black, steps_white, qs_white, ucbs_white, label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2):
    figsize = (10, 6.5)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 16
    fontsize_labels = 18
    fontsize_legend = 14
    grid_color = (0.4, 0.4, 0.4) 
    grid_dashes = (4.0, 4.0)
    legend_loc = "best"
    legend_handlelength = 4
    legend_labelspacing = 0.1
    alpha_ucb=0.25
    markersize = 3
    plt.figure(figsize=figsize)
    if title_1:
        plt.suptitle(title_1, fontsize=fontsize_suptitle)
    if title_2:
        plt.title(title_2, fontsize=fontsize_title)
    markers = {"marker": "o", "markersize": markersize}
    plt.plot(steps_black, qs_black, label=label_qs_black, color="red", **markers)    
    plt.fill_between(steps_black, qs_black, ucbs_black, color="red", alpha=alpha_ucb, label=label_ucbs_black)
    plt.plot(steps_white, qs_white, label=label_qs_white, color="blue", **markers)    
    plt.fill_between(steps_white, qs_white, ucbs_white, color="blue", alpha=0.25, label=label_ucbs_white)
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.legend(loc=legend_loc, prop={"size": fontsize_legend}, handlelength=legend_handlelength, labelspacing=legend_labelspacing)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)  
    plt.tight_layout(pad=0.4) 
    plt.show()
    
def best_action_plot_generator(experiments_hs, game_index, label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2):
    experiment_info = unzip_and_load_experiment(experiments_hs, FOLDER_EXPERIMENTS)
    moves_rounds = experiment_info["games_infos"][str(game_index)]["moves_rounds"]
    n_rounds = len(moves_rounds)
    steps_black = []
    qs_black = []
    ucbs_black = []
    steps_white = [] 
    qs_white = []
    ucbs_white = []
    for m in range(n_rounds):
        mr = moves_rounds[str(m + 1)]
        steps_black.append(m + 1)
        qs_black.append(mr["black_best_action_info"]["q"])
        ucbs_black.append(mr["black_best_action_info"]["ucb"])
        if "white_best_action_info" in mr:
            steps_white.append(m + 1.5)
            qs_white.append(mr["white_best_action_info"]["q"])
            ucbs_white.append( mr["white_best_action_info"]["ucb"])
    best_action_plot(steps_black, qs_black, ucbs_black, steps_white, qs_white, ucbs_white, label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2)

if __name__ == "__main__":        
    # scores_plot_ocp_thrifty_vs_vanilla()
    best_action_plot_generator("2249034921_44724_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_ocp_thrifty;C4_6x7;100]", 25, #11, 25, 47 
                               "BEST $\widehat{q}$ - MCTS-NC_1_INF_4_64_OCP_THRIFTY", "UCB - MCTS-NC_1_INF_4_64_OCP_THRIFTY", 
                               "BEST $\widehat{q}$ - MCTS_4_INF_VANILLA", "UCB - MCTS_4_INF_VANILLA", "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF CONNECT 4")
