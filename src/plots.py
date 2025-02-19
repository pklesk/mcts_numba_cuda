"""
Auxiliary module with utility functions for plots related MCTS-NC project and its experiments.

Link to project repository
--------------------------
`https://github.com/pklesk/mcts_numba_cuda <https://github.com/pklesk/mcts_numba_cuda>`_ 
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import numpy as np
from utils import unzip_and_load_experiment, dict_to_str

__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

FOLDER_EXPERIMENTS = "../experiments/"
FOLDER_EXTRAS = "../extras/"

def scores_array_plot(data, details, label_x, label_y, ticks_x, ticks_y, title):
    """Displays an array-like plot - a color map with averages of: scores, steps and depths - based on data from several experiments."""     
    figsize = (6, 6)
    fontsize_title = 21
    fontsize_ticks = 16
    fontsize_labels = 20
    fontsize_main = 20
    fontsize_details = 11       
    plt.figure(figsize=figsize)    
    
    mean_of_avgs = np.mean(data);
    title += f"\n[{mean_of_avgs * 100:.1f}% : {(1.0 - mean_of_avgs) * 100:.1f}%]"
    plt.title(title, fontsize=fontsize_title)
            
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
                             
    plt.tight_layout(pad=0.0) 
    plt.show()

def scores_array_plot_generator(experiments_hs_array, label_x, label_y, ticks_x, ticks_y, title, initial_player_flag=None):
    """Reads data from several experiments computes averages of scores, steps and depths and generates an array-like plot by calling ``scores_array_plot`` function."""
    print("SCORES-ARRAY-PLOT GENERATOR...")    
    data = np.zeros(experiments_hs_array.shape)
    details_playouts_steps = np.empty(experiments_hs_array.shape, dtype=object) 
    details_depths = np.empty(experiments_hs_array.shape, dtype=object)    
    ref_playouts = []
    ref_steps = []            
    ref_mean_depths = []
    ref_max_depths = []    
    for i in range(experiments_hs_array.shape[0]):
        for j in range(experiments_hs_array.shape[1]):
            experiment_info = unzip_and_load_experiment(experiments_hs_array[i, j], FOLDER_EXPERIMENTS)
            outcomes = []
            if False:
                data[i, j] = experiment_info["stats"]["score_b_mean"]
            n_games = experiment_info["matchup_info"]["n_games"]
            playouts = []
            steps = []            
            mean_depths = []
            max_depths = []
            for g in range(n_games):
                if initial_player_flag is not None:
                    if (initial_player_flag and g % 2 == 0) or (not initial_player_flag and g % 2 == 1):
                        continue
                outcome = int(experiment_info["games_infos"][str(g + 1)]["outcome"]) * 0.5 + 0.5
                if g % 2 == 0:
                    outcome = 1.0 - outcome
                outcomes.append(outcome)  
                moves_rounds = experiment_info["games_infos"][str(g + 1)]["moves_rounds"]
                main_player_prefix = "white_" if g % 2 == 0 else "black_"
                ref_player_prefix = "white_" if g % 2 == 1 else "black_"
                for m in range(len(moves_rounds)):
                    moves_round = moves_rounds[str(m + 1)]
                    mppi = main_player_prefix + "performance_info"
                    if mppi in moves_round:
                        playouts.append(moves_round[mppi]["playouts"])
                        steps.append(moves_round[mppi]["steps"])
                        trees_key = "trees" if "trees" in moves_round[mppi] else "tree"
                        mean_depths.append(moves_round[mppi][trees_key]["mean_depth"])
                        max_depths.append(moves_round[mppi][trees_key]["max_depth"])
                    rppi = ref_player_prefix + "performance_info"
                    if rppi in moves_round:
                        ref_playouts.append(moves_round[rppi]["playouts"])
                        ref_steps.append(moves_round[rppi]["steps"])
                        trees_key = "trees" if "trees" in moves_round[rppi] else "tree"
                        ref_mean_depths.append(moves_round[rppi][trees_key]["mean_depth"])
                        ref_max_depths.append(moves_round[rppi][trees_key]["max_depth"])                    
            details_playouts_steps[i, j] = f"{np.mean(playouts) / 10**6:.2f}M/{np.mean(steps) / 10**3:.2f}k"
            details_depths[i, j] = f"{np.mean(mean_depths):.1f}/{np.mean(max_depths):.1f}"
            data[i, j] = np.mean(outcomes)
    details = [details_playouts_steps, details_depths] 
    print(f"[reference player details: {np.mean(ref_playouts)}/{np.mean(ref_steps)}; {np.mean(ref_mean_depths)}/{np.mean(ref_max_depths)}]")
    print("SCORES-ARRAY-PLOT GENERATOR DONE.")        
    scores_array_plot(data, details, label_x, label_y, ticks_x, ticks_y, title)
    
def scores_array_plot_ocp_thrifty_vs_vanilla_c4(initial_player_flag=None):
    """Generates an array-like plot - a color map with averages of: scores, steps and depths - based on data from experiments: ocp_thrifty vs vanilla (Connect 4)."""
    experiments_hs_array = np.array([
        ["3505711246_84404_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_ocp_thrifty;C4_6x7;100]",
         "2363630254_99540_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_ocp_thrifty;C4_6x7;100]",
         "2779060134_98220_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_ocp_thrifty;C4_6x7;100]",
         "2226899934_08164_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_ocp_thrifty;C4_6x7;100]"],
        ["1674136622_62772_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_ocp_thrifty;C4_6x7;100]",
         "0532055630_45204_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_ocp_thrifty;C4_6x7;100]",
         "2757986856_45550_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_ocp_thrifty;C4_6x7;100]",
         "2205826656_55494_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_ocp_thrifty;C4_6x7;100]"],
        ["2305954670_86804_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_ocp_thrifty;C4_6x7;100]",
         "1163873678_01940_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_ocp_thrifty;C4_6x7;100]",
         "2715840300_72914_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_ocp_thrifty;C4_6x7;100]",
         "2163680100_50154_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_thrifty;C4_6x7;100]"],
        ["3569590766_67572_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_ocp_thrifty;C4_6x7;100]",
         "2427509774_50004_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_ocp_thrifty;C4_6x7;100]",
         "2631547188_62234_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_ocp_thrifty;C4_6x7;100]",
         "2079386988_72178_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_ocp_thrifty;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "OCP-THRIFTY (1$\,$s) vs VANILLA (5$\,$s)", initial_player_flag)    

def scores_array_plot_ocp_prodigal_vs_vanilla_c4(initial_player_flag=None):
    """Generates an array-like plot - a color map with averages of: scores, steps and depths - based on data from experiments: ocp_prodigal vs vanilla (Connect 4)."""    
    experiments_hs_array = np.array([
        ["3195982906_82784_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_ocp_prodigal;C4_6x7;100]",
         "0037595196_43010_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_ocp_prodigal;C4_6x7;100]",
         "0224716614_77068_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_ocp_prodigal;C4_6x7;100]",
         "3595406278_71788_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_ocp_prodigal;C4_6x7;100]"],
        ["3174909628_30114_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_ocp_prodigal;C4_6x7;100]",
         "0016521918_90340_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_ocp_prodigal;C4_6x7;100]",
         "1969956518_67564_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_ocp_prodigal;C4_6x7;100]",
         "1045678886_62284_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_ocp_prodigal;C4_6x7;100]"],
        ["3132763072_24774_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_ocp_prodigal;C4_6x7;100]",         
         "4269342658_85000_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_ocp_prodigal;C4_6x7;100]",
         "1165469030_15852_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_ocp_prodigal;C4_6x7;100]",
         "0241191398_10572_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]"],
        ["3048469960_46798_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_ocp_prodigal;C4_6x7;100]",
         "4185049546_07024_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_ocp_prodigal;C4_6x7;100]",
         "3851461350_45132_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_ocp_prodigal;C4_6x7;100]",
         "2927183718_39852_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_ocp_prodigal;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "OCP-PRODIGAL (1$\,$s) vs VANILLA (5$\,$s)", initial_player_flag)

def scores_array_plot_acp_thrifty_vs_vanilla_c4(initial_player_flag=None):
    """Generates an array-like plot - a color map with averages of: scores, steps and depths - based on data from experiments: acp_thrifty vs vanilla (Connect 4)."""
    experiments_hs_array = np.array([
        ["0225178830_78932_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_acp_thrifty;C4_6x7;100]",
         "3378065134_94068_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_acp_thrifty;C4_6x7;100]",
         "1096823050_67120_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_acp_thrifty;C4_6x7;100]",         
         "0544662850_44360_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_acp_thrifty;C4_6x7;100]"],
        ["2688571502_57300_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_acp_thrifty;C4_6x7;100]",
         "1546490510_39732_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_acp_thrifty;C4_6x7;100]",
         "1075749772_14450_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_acp_thrifty;C4_6x7;100]",
         "0523589572_91690_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_acp_thrifty;C4_6x7;100]"],
        ["3320389550_81332_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_acp_thrifty;C4_6x7;100]",         
         "2178308558_96468_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_acp_thrifty;C4_6x7;100]",
         "1033603216_09110_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_acp_thrifty;C4_6x7;100]",
         "0481443016_19054_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_acp_thrifty;C4_6x7;100]"],
        ["0289058350_62100_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_acp_thrifty;C4_6x7;100]",
         "3441944654_44532_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_acp_thrifty;C4_6x7;100]",
         "0949310104_31134_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_acp_thrifty;C4_6x7;100]",
         "0397149904_08374_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_acp_thrifty;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "ACP-THRIFTY (1$\,$s) vs VANILLA (5$\,$s)", initial_player_flag)

def scores_array_plot_acp_prodigal_vs_vanilla_c4(initial_player_flag=None):
    """Generates an array-like plot - a color map with averages of: scores, steps and depths - based on data from experiments: acp_prodigal vs vanilla (Connect 4)."""        
    experiments_hs_array = np.array([
        ["2586240854_62716_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_acp_prodigal;C4_6x7;100]",
         "3722820440_90238_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_acp_prodigal;C4_6x7;100]",
         "2273463942_77772_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_acp_prodigal;C4_6x7;100]",
         "1349186310_72492_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_acp_prodigal;C4_6x7;100]"],
        ["2565167576_10046_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_acp_prodigal;C4_6x7;100]",
         "3701747162_70272_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_acp_prodigal;C4_6x7;100]",
         "4018703846_35564_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_acp_prodigal;C4_6x7;100]",
         "3094426214_62988_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_acp_prodigal;C4_6x7;100]"],
        ["2523021020_04706_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_acp_prodigal;C4_6x7;100]",                  
         "3659600606_64932_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_acp_prodigal;C4_6x7;100]",
         "3214216358_16556_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_acp_prodigal;C4_6x7;100]",
         "2289938726_11276_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_acp_prodigal;C4_6x7;100]"],
        ["2438727908_26730_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_acp_prodigal;C4_6x7;100]",
         "3575307494_54252_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_acp_prodigal;C4_6x7;100]",
         "1605241382_13132_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_acp_prodigal;C4_6x7;100]",
         "0680963750_07852_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_acp_prodigal;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "ACP-PRODIGAL (1$\,$s) vs VANILLA (5$\,$s)", initial_player_flag)    

def best_action_plot(moves_rounds_black, qs_black, ucbs_black, moves_rounds_white, qs_white, ucbs_white, 
                     label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2,
                     ucbs_factor=1.0, ucbs_black_color=None, ucbs_white_color=None):
    """Displays plot of estimates on best actions' values (and their UCBs) along a given game."""    
    figsize = (10.0, 5.0)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 8.5
    fontsize_labels = 18
    fontsize_legend = 11
    grid_color = (0.4, 0.4, 0.4) 
    grid_dashes = (4.0, 4.0)
    legend_loc = "best" # "upper left"
    legend_handlelength = 4
    legend_labelspacing = 0.1
    alpha_ucb=0.25
    markersize = 3
    plt.figure(figsize=figsize)
    if title_1:
        plt.suptitle(title_1, fontsize=fontsize_suptitle)
    if title_2:
        plt.title(title_2, fontsize=fontsize_title)
    if ucbs_black_color is None:
        ucbs_black_color = "red"
    if ucbs_white_color is None:
        ucbs_white_color = "blue"                
    markers = {"marker": "o", "markersize": markersize}
    plt.plot(moves_rounds_black, qs_black, label=label_qs_black, color="red", **markers)    
    ucbs_black = ucbs_factor * (np.array(ucbs_black) - np.array(qs_black)) + np.array(qs_black)     
    plt.fill_between(moves_rounds_black, qs_black, ucbs_black, color=ucbs_black_color, alpha=alpha_ucb, label=label_ucbs_black)
    plt.plot(moves_rounds_white, qs_white, label=label_qs_white, color="blue", **markers)
    ucbs_white = ucbs_factor * (np.array(ucbs_white) - np.array(qs_white)) + np.array(qs_white)    
    plt.fill_between(moves_rounds_white, qs_white, ucbs_white, color=ucbs_white_color, alpha=0.25, label=label_ucbs_white)
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.legend(loc=legend_loc, prop={"size": fontsize_legend}, handlelength=legend_handlelength, labelspacing=legend_labelspacing)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(FixedLocator(np.arange(0, 1.125, 0.125)))
    ticks_x = np.arange(1, max(max(moves_rounds_black), max(moves_rounds_white)) + 1, 1)
    plt.xticks(ticks_x, fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)    
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)  
    plt.tight_layout(pad=0.4) 
    plt.show()
    
def best_action_plot_generator(experiments_hs, game_index, 
                               label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2,
                               ucbs_factor=1.0, ucbs_black_color=None, ucbs_white_color=None):
    """Reads data from an experiment and generates a 'best action plot' by calling ``best_action_plot`` function."""
    print("BEST-ACTION-PLOT GENERATOR...") 
    experiment_info = unzip_and_load_experiment(experiments_hs, FOLDER_EXPERIMENTS)
    moves_rounds = experiment_info["games_infos"][str(game_index)]["moves_rounds"]
    n_rounds = len(moves_rounds)
    moves_rounds_black = []
    qs_black = []
    ucbs_black = []
    moves_rounds_white = [] 
    qs_white = []
    ucbs_white = []
    for m in range(n_rounds):
        mr = moves_rounds[str(m + 1)]
        moves_rounds_black.append(m + 1)
        qs_black.append(mr["black_best_action_info"]["q"])
        ucbs_black.append(mr["black_best_action_info"]["ucb"])
        if "white_best_action_info" in mr:
            moves_rounds_white.append(m + 1.5)
            qs_white.append(mr["white_best_action_info"]["q"])
            ucbs_white.append( mr["white_best_action_info"]["ucb"])
    print("BEST-ACTION-PLOT GENERATOR DONE.")
    best_action_plot(moves_rounds_black, qs_black, ucbs_black, moves_rounds_white, qs_white, ucbs_white, 
                     label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2, ucbs_factor, ucbs_black_color, ucbs_white_color)    

def depths_plot(moves_rounds_black, mean_depths_black, max_depths_black, moves_rounds_white, mean_depths_white, max_depths_white, 
                label_mean_depths_black, label_max_depths_black, label_mean_depths_white, label_max_depths_white, label_x, label_y, title_1, title_2):
    """Displays plot of averages of reached depths (mean and maximum) along a given game."""    
    figsize = (10, 5.0)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 8.5
    fontsize_labels = 18
    fontsize_legend = 11
    grid_color = (0.4, 0.4, 0.4) 
    grid_dashes = (4.0, 4.0)
    legend_loc = "best" # "lower left"
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
    plt.plot(moves_rounds_black, mean_depths_black, label=label_mean_depths_black, color="red", **markers)    
    plt.plot(moves_rounds_black, max_depths_black, label=label_max_depths_black, color="red", **markers, linestyle="--")    
    plt.plot(moves_rounds_white, mean_depths_white, label=label_mean_depths_white, color="blue", **markers)    
    plt.plot(moves_rounds_white, max_depths_white, label=label_max_depths_white, color="blue", **markers, linestyle="--")
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.legend(loc=legend_loc, prop={"size": fontsize_legend}, handlelength=legend_handlelength, labelspacing=legend_labelspacing)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))    
    ticks_x = np.arange(1, max(max(moves_rounds_black), max(moves_rounds_white)) + 1, 1)
    plt.xticks(ticks_x, fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)        
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)  
    plt.tight_layout(pad=0.4)
    plt.show()

def depths_plot_generator(experiments_hs, game_index, 
                          label_mean_depths_black, label_max_depths_black, label_mean_depths_white, label_max_depths_white, label_x, label_y, title_1, title_2):
    """Reads data from an experiment and generates a 'depths plot' by calling ``best_action_plot`` function."""    
    experiment_info = unzip_and_load_experiment(experiments_hs, FOLDER_EXPERIMENTS)
    moves_rounds = experiment_info["games_infos"][str(game_index)]["moves_rounds"]
    n_rounds = len(moves_rounds)
    moves_rounds_black = []
    mean_depths_black = []
    max_depths_black = []
    moves_rounds_white = []     
    mean_depths_white = []
    max_depths_white = []
    for m in range(n_rounds):
        mr = moves_rounds[str(m + 1)]
        moves_rounds_black.append(m + 1)
        trees_key = "trees" if "trees" in mr["black_performance_info"] else "tree"        
        mean_depths_black.append(mr["black_performance_info"][trees_key]["mean_depth"])
        max_depths_black.append(mr["black_performance_info"][trees_key]["max_depth"])
        if "white_best_action_info" in mr:
            moves_rounds_white.append(m + 1.5)
            trees_key = "trees" if "trees" in mr["white_performance_info"] else "tree"        
            mean_depths_white.append(mr["white_performance_info"][trees_key]["mean_depth"])
            max_depths_white.append(mr["white_performance_info"][trees_key]["max_depth"])
    depths_plot(moves_rounds_black, mean_depths_black, max_depths_black, moves_rounds_white, mean_depths_white, max_depths_white, 
                label_mean_depths_black, label_max_depths_black, label_mean_depths_white, label_max_depths_white, label_x, label_y, title_1, title_2)

def averages_printout_generator(experiments_hs_array, ai_instance_name):
    """Prints out averages of: playouts / steps and mean / maximum depths for a given series of experiments.""" 
    print("AVERAGES PRINTOUT...")
    playouts = []
    steps = []            
    mean_depths = []
    max_depths = []            
    for i in range(experiments_hs_array.shape[0]):
        experiment_info = unzip_and_load_experiment(experiments_hs_array[i], FOLDER_EXPERIMENTS)
        n_games = experiment_info["matchup_info"]["n_games"]
        for g in range(n_games):            
            main_player_prefix = "white_" if experiment_info["games_infos"][str(g + 1)]["white"] == ai_instance_name else "black_"            
            moves_rounds = experiment_info["games_infos"][str(g + 1)]["moves_rounds"]
            for m in range(len(moves_rounds)):
                moves_round = moves_rounds[str(m + 1)]
                mppi = main_player_prefix + "performance_info"
                if mppi in moves_round:
                    playouts.append(moves_round[mppi]["playouts"])
                    steps.append(moves_round[mppi]["steps"])
                    trees_key = "trees" if "trees" in moves_round[mppi] else "tree"
                    mean_depths.append(moves_round[mppi][trees_key]["mean_depth"])
                    max_depths.append(moves_round[mppi][trees_key]["max_depth"])
    print(f"THE AVERAGES -> PLAYOUTS/STEPS: {np.mean(playouts)}/{np.mean(steps)}, MEAN DEPTH/MAX DEPTH: {np.mean(mean_depths)}/{np.mean(max_depths)}")
    print("AVERAGES PRINTOUT GENERATOR DONE.")

def averages_printout_5s_vanilla_c4():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 5s vanilla (Connect 4)."""    
    averages_printout_generator(np.array([
        "2959981740_01490_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_128_ocp_thrifty;C4_6x7;100]",
        "2749967598_89204_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "0725584456_47630_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "0503747630_89908_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTS(search_time_limit=5.0, search_steps_limit=inf, vanilla=True, ucb_c=2.0, seed: 0)")    
    
def averages_printout_5s_ocp_thrifty_c4():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 5s ocp_thrifty (Connect 4)."""    
    averages_printout_generator(np.array([
        "2959981740_01490_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_128_ocp_thrifty;C4_6x7;100]",
        "1311471072_93670_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "3070453690_29088_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "3360218400_94374_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=128, variant='ocp_thrifty', device_memory=2.0, ucb_c=2.0, seed: 0)")  

def averages_printout_5s_ocp_prodigal_c4():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 5s ocp_prodigal (Connect 4)."""    
    averages_printout_generator(np.array([
        "2749967598_89204_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "1311471072_93670_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "0995822742_91740_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "2504702716_35906_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='ocp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)")
    
def averages_printout_5s_acp_thrifty_c4():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 5s acp_thrifty (Connect 4)."""    
    averages_printout_generator(np.array([        
        "0725584456_47630_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "3070453690_29088_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "0995822742_91740_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "3763533572_41898_048_[mctsnc_5_inf_4_256_acp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_thrifty', device_memory=2.0, ucb_c=2.0, seed: 0)")    

def averages_printout_5s_acp_prodigal_c4():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 5s acp_prodigal (Connect 4)."""    
    averages_printout_generator(np.array([
        "0503747630_89908_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]",
        "3360218400_94374_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]",
        "2504702716_35906_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]",
        "3763533572_41898_048_[mctsnc_5_inf_4_256_acp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)")

def averages_printout_30s_vanilla_gomoku():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 30s vanilla (Gomoku)."""
    averages_printout_generator(np.array([
        "3014955156_02650_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_128_ocp_thrifty_16g;Gomoku_15x15;100]",
        "1681612016_34230_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "4070724948_30746_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "3240654036_09850_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTS(search_time_limit=30.0, search_steps_limit=inf, vanilla=True, ucb_c=2.0, seed: 0)")

def averages_printout_30s_ocp_thrifty_gomoku():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 30s ocp_thrifty (Gomoku)."""
    averages_printout_generator(np.array([
        "3014955156_02650_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_128_ocp_thrifty_16g;Gomoku_15x15;100]",
        "4039933000_53870_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "2602789548_96434_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "1304007724_29490_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=128, variant='ocp_thrifty', device_memory=16.0, ucb_c=2.0, seed: 0)")

def averages_printout_30s_ocp_prodigal_gomoku():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 30s ocp_prodigal (Gomoku)."""
    averages_printout_generator(np.array([
        "1681612016_34230_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "4039933000_53870_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "1988707178_17328_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "3876337002_15280_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='ocp_prodigal', device_memory=16.0, ucb_c=2.0, seed: 0)")

def averages_printout_30s_acp_thrifty_gomoku():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 30s acp_thrifty (Gomoku)."""
    averages_printout_generator(np.array([
        "4070724948_30746_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "2602789548_96434_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "1988707178_17328_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_thrifty', device_memory=16.0, ucb_c=2.0, seed: 0)")

def averages_printout_30s_acp_prodigal_gomoku():
    """Prints out averages of: playouts / steps and mean / maximum depths for experiments involving: 30s acp_prodigal (Gomoku)."""
    averages_printout_generator(np.array([
        "3240654036_09850_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]",
        "1304007724_29490_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]",
        "3876337002_15280_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]",
        "2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_prodigal', device_memory=16.0, ucb_c=2.0, seed: 0)")
    
def playouts_per_second_plot(n_plots, label_x, label_y, ticks_x, n_trees_values, title, label_prefix, label_suffix, data_pps, ref_label, ref_pps_avg):
    """Displays multiple plots of 'playouts per second' quantity, averaged over multiple Connect 4 games."""
    figsize = (10.0, 5.0)        
    fontsize_title = 21
    fontsize_ticks = 8.5
    fontsize_labels = 18
    fontsize_legend = 11
    grid_color = (0.4, 0.4, 0.4) 
    grid_dashes = (4.0, 4.0)
    legend_loc = "best" # "upper left"
    legend_handlelength = 4
    legend_labelspacing = 0.1
    alpha_ucb=0.25
    markersize = 3
    plt.figure(figsize=figsize)
    if title:
        plt.title(title, fontsize=fontsize_title)
    markers = {"marker": "o", "markersize": markersize}    
    max_gray = 0.75
    for i in range(n_plots - 1, -1, -1):
        gray_value = max_gray - (i / (n_plots - 1)) * max_gray 
        plt.plot(ticks_x, data_pps[i], color=str(gray_value), label=f"{label_prefix}_{n_trees_values[i]}_m_{label_suffix}", **markers)
    plt.plot(ticks_x, [ref_pps_avg] * n_plots, label=ref_label, color=str(max_gray), linestyle="--", **markers)
    plt.yscale("log")
    plt.ylim(1e3, 1e8)
    plt.xticks(ticks_x, fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)    
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.legend(loc="upper left", prop={"size": fontsize_legend}, handlelength=legend_handlelength, labelspacing=legend_labelspacing)  
    plt.tight_layout(pad=0.4) 
    plt.show()    

def playouts_per_second_plot_generator(experiments_hs_array, label_x, label_y, ticks_x, n_trees_values, title, label_prefix, label_suffix, ref_label):
    """Reads data from several experiments, computes averages of 'playouts per second' quantity, and generates a plot by calling ``playouts_per_second_plot`` function."""
    print("PLAYOUTS-PER-SECOND-PLOT GENERATOR...")    
    data_pps = np.zeros(experiments_hs_array.shape)     
    ref_pps = []
    for i in range(experiments_hs_array.shape[0]):
        for j in range(experiments_hs_array.shape[1]):
            experiment_info = unzip_and_load_experiment(experiments_hs_array[i, j], FOLDER_EXPERIMENTS)
            n_games = experiment_info["matchup_info"]["n_games"]
            pps = []
            for g in range(n_games):
                moves_rounds = experiment_info["games_infos"][str(g + 1)]["moves_rounds"]
                main_player_prefix = "white_" if g % 2 == 0 else "black_"
                ref_player_prefix = "white_" if g % 2 == 1 else "black_"
                for m in range(len(moves_rounds)):
                    moves_round = moves_rounds[str(m + 1)]
                    mppi = main_player_prefix + "performance_info"
                    if mppi in moves_round:
                        pps.append(moves_round[mppi]["playouts_per_second"])
                    rppi = ref_player_prefix + "performance_info"
                    if rppi in moves_round:
                        ref_pps.append(moves_round[rppi]["playouts_per_second"])
            data_pps[i, j] = np.mean(pps) 
    ref_pps_avg = np.mean(ref_pps)
    print(f"[reference player playouts per second: {ref_pps_avg}]")
    print("PLAYOUTS-PER-SECOND-PLOT GENERATOR DONE.")
    playouts_per_second_plot(experiments_hs_array.shape[0], label_x, label_y, ticks_x, n_trees_values, title, label_prefix, label_suffix, data_pps, ref_label, ref_pps_avg)

def playouts_per_second_plot_acp_prodigal_vs_vanilla_c4():
    """Generates a series of plots with 'playouts per second' quantity (in logarithmic scale) based on data from experiments: acp_prodigal vs vanilla (Connect 4)."""        
    experiments_hs_array = np.array([
        ["2586240854_62716_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_acp_prodigal;C4_6x7;100]",
         "3722820440_90238_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_acp_prodigal;C4_6x7;100]",
         "2273463942_77772_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_acp_prodigal;C4_6x7;100]",
         "1349186310_72492_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_acp_prodigal;C4_6x7;100]"],
        ["2565167576_10046_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_acp_prodigal;C4_6x7;100]",
         "3701747162_70272_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_acp_prodigal;C4_6x7;100]",
         "4018703846_35564_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_acp_prodigal;C4_6x7;100]",
         "3094426214_62988_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_acp_prodigal;C4_6x7;100]"],
        ["2523021020_04706_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_acp_prodigal;C4_6x7;100]",                  
         "3659600606_64932_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_acp_prodigal;C4_6x7;100]",
         "3214216358_16556_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_acp_prodigal;C4_6x7;100]",
         "2289938726_11276_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_acp_prodigal;C4_6x7;100]"],
        ["2438727908_26730_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_acp_prodigal;C4_6x7;100]",
         "3575307494_54252_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_acp_prodigal;C4_6x7;100]",
         "1605241382_13132_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_acp_prodigal;C4_6x7;100]",
         "0680963750_07852_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_acp_prodigal;C4_6x7;100]"]
        ])
    playouts_per_second_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "AVGS. OF PLAYOUTS PER SECOND", [32, 64, 128, 256], [1, 2, 4, 8], 
                                       "CONNECT 4: ACP-PRODIGAL (1$\,$s) vs VANILLA (5$\,$s)", "MCTS-NC_1_INF", "ACP_PRODIGAL", "MCTS_5_INF_VANILLA (REFERENCE)")    

def playouts_per_second_plot_ocp_prodigal_vs_vanilla_c4():
    """Generates a series of plots with 'playouts per second' quantity (in logarithmic scale) based on data from experiments: ocp_prodigal vs vanilla (Connect 4)."""    
    experiments_hs_array = np.array([
        ["3195982906_82784_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_ocp_prodigal;C4_6x7;100]",
         "0037595196_43010_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_ocp_prodigal;C4_6x7;100]",
         "0224716614_77068_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_ocp_prodigal;C4_6x7;100]",
         "3595406278_71788_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_ocp_prodigal;C4_6x7;100]"],
        ["3174909628_30114_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_ocp_prodigal;C4_6x7;100]",
         "0016521918_90340_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_ocp_prodigal;C4_6x7;100]",
         "1969956518_67564_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_ocp_prodigal;C4_6x7;100]",
         "1045678886_62284_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_ocp_prodigal;C4_6x7;100]"],
        ["3132763072_24774_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_ocp_prodigal;C4_6x7;100]",         
         "4269342658_85000_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_ocp_prodigal;C4_6x7;100]",
         "1165469030_15852_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_ocp_prodigal;C4_6x7;100]",
         "0241191398_10572_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]"],
        ["3048469960_46798_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_ocp_prodigal;C4_6x7;100]",
         "4185049546_07024_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_ocp_prodigal;C4_6x7;100]",
         "3851461350_45132_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_ocp_prodigal;C4_6x7;100]",
         "2927183718_39852_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_ocp_prodigal;C4_6x7;100]"]
        ])
    playouts_per_second_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "AVGS. OF PLAYOUTS PER SECOND", [32, 64, 128, 256], [1, 2, 4, 8], 
                                       "CONNECT 4: OCP-PRODIGAL (1$\,$s) vs VANILLA (5$\,$s)", "MCTS-NC_1_INF", "OCP_PRODIGAL", "MCTS_5_INF_VANILLA (REFERENCE)")
    
def stats_detailed_printout(experiment_hs):
    """Prints out stats (with side-dependency distinction) for a given experiment.""" 
    print("STATS DETAILED PRINTOUT...")
    a_first_outcomes = []
    a_second_outcomes = []
    b_first_outcomes = []
    b_second_outcomes = []    
    experiment_info = unzip_and_load_experiment(experiment_hs, FOLDER_EXPERIMENTS)
    print("MATCH-UP INFO:\n" + dict_to_str(experiment_info["matchup_info"]))
    print("STATS:\n" + dict_to_str(experiment_info["stats"]))
    n_games = experiment_info["matchup_info"]["n_games"]
    for g in range(n_games):            
        outcome = int(experiment_info["games_infos"][str(g + 1)]["outcome"]) * 0.5 + 0.5
        if g % 2 == 0:            
            a_first_outcomes.append(outcome)
            b_second_outcomes.append(1.0 - outcome)
        else:
            a_second_outcomes.append(1.0 - outcome)
            b_first_outcomes.append(outcome)            
    print(f"SIDE A STATS DETAILED -> FIRST: {np.mean(a_first_outcomes)}, SECOND: {np.mean(a_second_outcomes)}")
    print(f"SIDE B STATS DETAILED -> FIRST: {np.mean(b_first_outcomes)}, SECOND: {np.mean(b_second_outcomes)}")
    print("STATS DETAILED PRINTOUT DONE.")    
    
if __name__ == "__main__":
    print("PLOTS FOR MCTS-NC EXPERIMENTS...")
    
    # scores_array_plot_ocp_thrifty_vs_vanilla_c4()
    
    # scores_array_plot_ocp_prodigal_vs_vanilla_c4()
    
    # scores_array_plot_acp_thrifty_vs_vanilla_c4()
        
    # scores_array_plot_acp_prodigal_vs_vanilla_c4()

    best_action_plot_generator("0241191398_10572_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]", 9,
                               "BEST $\widehat{q}$ - MCTS_5_INF_VANILLA", "UCB - MCTS_5_INF_VANILLA",      
                               "BEST $\widehat{q}$ - MCTS-NC_1_INF_4_256_OCP_PRODIGAL", "UCB - MCTS-NC_1_INF_4_256_OCP_PRODIGAL",     
                               "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF CONNECT 4 (NO. 9/100)")    
    depths_plot_generator("0241191398_10572_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]", 9,  
                          "MEAN DEPTHS - MCTS_5_INF_VANILLA", "MAX DEPTHS - MCTS_5_INF_VANILLA",
                          "MEAN DEPTHS - MCTS-NC_1_INF_4_256_OCP_PRODIGAL", "MAX DEPTHS - MCTS-NC_1_INF_4_256_OCP_PRODIGAL",                                  
                          "MOVES ROUND", "MEAN, MAXIMUM DEPTHS", None, "SAMPLE GAME OF CONNECT 4 (NO. 9/100)")
    
    # best_action_plot_generator("2504702716_35906_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]", 57,
    #                            "BEST $\widehat{q}$ - MCTS-NC_5_INF_4_256_OCP_PRODIGAL", "10 x UCB - MCTS-NC_5_INF_4_256_OCP_PRODIGAL",      
    #                            "BEST $\widehat{q}$ - MCTS-NC_5_INF_4_256_ACP_PRODIGAL", "10 x UCB - MCTS-NC_5_INF_4_256_ACP_PRODIGAL",     
    #                            "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF CONNECT 4 (NO. 57/100)", 10.0)    
    # depths_plot_generator("2504702716_35906_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]", 57,  
    #                       "BEST $\widehat{q}$ - MCTS-NC_5_INF_4_256_OCP_PRODIGAL", "UCB - MCTS-NC_5_INF_4_256_OCP_PRODIGAL",      
    #                       "BEST $\widehat{q}$ - MCTS-NC_5_INF_4_256_ACP_PRODIGAL", "UCB - MCTS-NC_5_INF_4_256_ACP_PRODIGAL",                                  
    #                       "MOVES ROUND", "MEAN, MAXIMUM DEPTHS", None, "SAMPLE GAME OF CONNECT 4 (NO. 57/100)")    
    
    # best_action_plot_generator("2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]", 11,
    #                            "BEST $\widehat{q}$ - MCTS-NC_30_INF_4_256_ACP_THRIFTY", "25 x UCB - MCTS-NC_30_INF_4_256_ACP_THRIFTY",
    #                            "BEST $\widehat{q}$ - MCTS-NC_30_INF_4_256_ACP_PRODIGAL", "25 x UCB - MCTS-NC_30_INF_4_256_ACP_PRODIGAL",                                  
    #                            "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF GOMOKU (NO. 11/100)", 25.0)    
    # depths_plot_generator("2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]", 11,  
    #                       "MEAN DEPTHS - MCTS-NC_30_INF_4_256_ACP_THRIFTY", "MAX DEPTHS - MCTS-NC_30_INF_4_256_ACP_THRIFTY",
    #                       "MEAN DEPTHS - MCTS-NC_30_INF_4_256_ACP_PRODIGAL", "MAX DEPTHS - MCTS-NC_30_INF_4_256_ACP_PRODIGAL",                                  
    #                       "MOVES ROUND", "MEAN, MAXIMUM DEPTHS", None, "SAMPLE GAME OF GOMOKU (NO. 11/100)")
        
    # averages_printout_5s_vanilla_c4()
    
    # averages_printout_5s_ocp_thrifty_c4()
    
    # averages_printout_5s_ocp_prodigal_c4()
    
    # averages_printout_5s_acp_thrifty_c4()
    
    # averages_printout_5s_acp_prodigal_c4()
    
    # averages_printout_30s_vanilla_gomoku()
    
    # averages_printout_30s_ocp_thrifty_gomoku()
    
    # averages_printout_30s_ocp_prodigal_gomoku()
    
    # averages_printout_30s_acp_thrifty_gomoku()
    
    # averages_printout_30s_acp_prodigal_gomoku()
    
    # playouts_per_second_plot_ocp_prodigal_vs_vanilla_c4()
    
    # playouts_per_second_plot_acp_prodigal_vs_vanilla_c4()
    
    # stats_detailed_printout("0503747630_89908_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]")
    
    print("PLOTS FOR MCTS-NC EXPERIMENTS DONE.")