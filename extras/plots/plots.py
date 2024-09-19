import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import numpy as np
from utils import unzip_and_load_experiment

FOLDER_EXPERIMENTS = "../../experiments/"

def scores_array_plot(data, details, label_x, label_y, ticks_x, ticks_y, title):    
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

def scores_array_plot_generator(experiments_hs_array, label_x, label_y, ticks_x, ticks_y, title):
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
            data[i, j] = experiment_info["stats"]["score_b_mean"]
            n_games = experiment_info["matchup_info"]["n_games"]
            playouts = []
            steps = []            
            mean_depths = []
            max_depths = []
            for g in range(n_games):
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
                        ref_mean_depths.append(moves_round[rppi]["tree"]["mean_depth"])
                        ref_max_depths.append(moves_round[rppi]["tree"]["max_depth"])                    
            details_playouts_steps[i, j] = f"{np.mean(playouts) / 10**6:.2f}M/{np.mean(steps) / 10**3:.2f}k"
            details_depths[i, j] = f"{np.mean(mean_depths):.1f}/{np.mean(max_depths):.1f}"
    details = [details_playouts_steps, details_depths]
    print(f"[reference player details: {np.mean(ref_playouts)}/{np.mean(ref_steps)}; {np.mean(ref_mean_depths)}/{np.mean(ref_max_depths)}]")
    print("SCORES-ARRAY-PLOT GENERATOR DONE.")        
    scores_array_plot(data, details, label_x, label_y, ticks_x, ticks_y, title)
    

def scores_array_plot_ocp_thrifty_vs_vanilla():
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
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "OCP-THRIFTY (1s) vs VANILLA (4s)")    

def scores_array_plot_ocp_prodigal_vs_vanilla():
    experiments_hs_array = np.array([
        ["1302514517_91136_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_32_ocp_prodigal;C4_6x7;100]",
         "2439094103_18658_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_64_ocp_prodigal;C4_6x7;100]",
         "0611358305_22348_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_128_ocp_prodigal;C4_6x7;100]",
         "3982047969_49772_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_256_ocp_prodigal;C4_6x7;100]"],
        ["1281441239_38466_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_32_ocp_prodigal;C4_6x7;100]",
         "2418020825_65988_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_64_ocp_prodigal;C4_6x7;100]",
         "2356598209_12844_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_128_ocp_prodigal;C4_6x7;100]",
         "1432320577_07564_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_256_ocp_prodigal;C4_6x7;100]"],
        ["1239294683_33126_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_32_ocp_prodigal;C4_6x7;100]",         
         "2375874269_93352_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_ocp_prodigal;C4_6x7;100]",
         "1552110721_61132_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_128_ocp_prodigal;C4_6x7;100]",
         "0627833089_55852_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]"],
        ["1155001571_55150_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_32_ocp_prodigal;C4_6x7;100]",
         "2291581157_82672_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_64_ocp_prodigal;C4_6x7;100]",
         "4238103041_90412_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_128_ocp_prodigal;C4_6x7;100]",
         "3313825409_85132_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_256_ocp_prodigal;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "OCP-PRODIGAL (1s) vs VANILLA (4s)")

def scores_array_plot_acp_thrifty_vs_vanilla():
    experiments_hs_array = np.array([
        ["1310340073_54420_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_32_acp_thrifty;C4_6x7;100]",
         "0168259081_36852_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_64_acp_thrifty;C4_6x7;100]",
         "3498321957_42768_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_128_acp_thrifty;C4_6x7;100]",         
         "2946161757_52712_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_256_acp_thrifty;C4_6x7;100]"],
        ["3773732745_00084_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_32_acp_thrifty;C4_6x7;100]",
         "2631651753_15220_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_64_acp_thrifty;C4_6x7;100]",
         "3477248679_90098_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_128_acp_thrifty;C4_6x7;100]",
         "2925088479_00042_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_256_acp_thrifty;C4_6x7;100]"],
        ["0110583497_56820_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_32_acp_thrifty;C4_6x7;100]",         
         "3263469801_39252_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_acp_thrifty;C4_6x7;100]",
         "3435102123_17462_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_128_acp_thrifty;C4_6x7;100]",
         "2882941923_94702_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_256_acp_thrifty;C4_6x7;100]"],
        ["1374219593_04884_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_32_acp_thrifty;C4_6x7;100]",
         "0232138601_20020_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_64_acp_thrifty;C4_6x7;100]",
         "3350809011_06782_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_128_acp_thrifty;C4_6x7;100]",
         "2798648811_16726_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_256_acp_thrifty;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "ACP-THRIFTY (1s) vs VANILLA (4s)")

def scores_array_plot_acp_prodigal_vs_vanilla():
    experiments_hs_array = np.array([
        ["0692772465_38364_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_32_acp_prodigal;C4_6x7;100]",
         "1829352051_98590_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_64_acp_prodigal;C4_6x7;100]",
         "2660105633_23052_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_128_acp_prodigal;C4_6x7;100]",
         "1735828001_17772_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_256_acp_prodigal;C4_6x7;100]"],
        ["0671699187_85694_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_32_acp_prodigal;C4_6x7;100]",
         "1808278773_45920_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_64_acp_prodigal;C4_6x7;100]",
         "0110378241_13548_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_128_acp_prodigal;C4_6x7;100]",
         "3481067905_08268_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_256_acp_prodigal;C4_6x7;100]"],
        ["0629552631_13058_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_32_acp_prodigal;C4_6x7;100]",         
         "1766132217_73284_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_acp_prodigal;C4_6x7;100]",
         "3600858049_61836_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_128_acp_prodigal;C4_6x7;100]",
         "2676580417_56556_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_256_acp_prodigal;C4_6x7;100]"],
        ["0545259519_02378_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_32_acp_prodigal;C4_6x7;100]",
         "1681839105_62604_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_64_acp_prodigal;C4_6x7;100]",
         "1991883073_58412_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_128_acp_prodigal;C4_6x7;100]",
         "1067605441_53132_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_256_acp_prodigal;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "ACP-PRODIGAL (1s) vs VANILLA (4s)")


def scores_curve_plot(data, label_x, label_y, ticks_x, title_1, title_2):    
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
    for i in range(data.shape[0]):
        plt.plot(data[i])    
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.xticks(ticks=np.arange(data.shape[1]), labels=ticks_x, fontsize=fontsize_ticks)     
    plt.tight_layout(pad=0.4) 
    plt.show()

def scores_curve_plot_generator(experiments_hs_array, label_x, label_y, ticks_x, title_1, title_2):
    print("SCORES-CURVE-PLOT GENERATOR...")    
    data = np.zeros(experiments_hs_array.shape)
    for i in range(experiments_hs_array.shape[0]):
        for j in range(experiments_hs_array.shape[1]):
            experiment_info = unzip_and_load_experiment(experiments_hs_array[i, j], FOLDER_EXPERIMENTS)
            data[i, j] = experiment_info["stats"]["score_b_mean"]
            steps = []
            mean_depths = []
            max_depths = []                    
    print("SCORES-CURVE-PLOT GENERATOR DONE.")
    scores_curve_plot(data, label_x, label_y, ticks_x, title_1, title_2)    

def best_action_plot(moves_rounds_black, qs_black, ucbs_black, moves_rounds_white, qs_white, ucbs_white, 
                     label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2):
    figsize = (10, 6.0)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 13
    fontsize_labels = 18
    fontsize_legend = 13
    grid_color = (0.4, 0.4, 0.4) 
    grid_dashes = (4.0, 4.0)
    legend_loc = "upper left"
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
    plt.plot(moves_rounds_black, qs_black, label=label_qs_black, color="red", **markers)    
    plt.fill_between(moves_rounds_black, qs_black, ucbs_black, color="red", alpha=alpha_ucb, label=label_ucbs_black)
    plt.plot(moves_rounds_white, qs_white, label=label_qs_white, color="blue", **markers)    
    plt.fill_between(moves_rounds_white, qs_white, ucbs_white, color="blue", alpha=0.25, label=label_ucbs_white)
    plt.xlabel(label_x, fontsize=fontsize_labels)
    plt.ylabel(label_y, fontsize=fontsize_labels)
    plt.legend(loc=legend_loc, prop={"size": fontsize_legend}, handlelength=legend_handlelength, labelspacing=legend_labelspacing)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)    
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)  
    plt.tight_layout(pad=0.4) 
    plt.show()
    
def best_action_plot_generator(experiments_hs, game_index, 
                               label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2):
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
    print("BEST-ACTION-PLOT GENERATOR DONE")
    best_action_plot(moves_rounds_black, qs_black, ucbs_black, moves_rounds_white, qs_white, ucbs_white, label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2)    

def depths_plot(moves_rounds_black, mean_depths_black, max_depths_black, moves_rounds_white, mean_depths_white, max_depths_white, 
                label_mean_depths_black, label_max_depths_black, label_mean_depths_white, label_max_depths_white, label_x, label_y, title_1, title_2):
    figsize = (10, 6.0)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 13
    fontsize_labels = 18
    fontsize_legend = 13
    grid_color = (0.4, 0.4, 0.4) 
    grid_dashes = (4.0, 4.0)
    legend_loc = "upper left"
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
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)        
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)  
    plt.tight_layout(pad=0.4) 
    plt.show()

def depths_plot_generator(experiments_hs, game_index, 
                          label_mean_depths_black, label_max_depths_black, label_mean_depths_white, label_max_depths_white, label_x, label_y, title_1, title_2):
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


if __name__ == "__main__":        
    
    # scores_array_plot_ocp_thrifty_vs_vanilla()
    
    # scores_array_plot_ocp_prodigal_vs_vanilla()
    
    scores_array_plot_acp_thrifty_vs_vanilla()
    
    # scores_array_plot_acp_prodigal_vs_vanilla()
    
    # best_action_plot_generator("2249034921_44724_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_ocp_thrifty;C4_6x7;100]", 25, #11, 25, 47
    #                            "BEST $\widehat{q}$ - MCTS_4_INF_VANILLA", "UCB - MCTS_4_INF_VANILLA",      
    #                            "BEST $\widehat{q}$ - MCTS-NC_1_INF_4_64_OCP_THRIFTY", "UCB - MCTS-NC_1_INF_4_64_OCP_THRIFTY",     
    #                            "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF CONNECT 4")
    
    # depths_plot_generator("2249034921_44724_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_ocp_thrifty;C4_6x7;100]", 25,  
    #                       "MEAN DEPTHS - MCTS_4_INF_VANILLA", "MAX DEPTHS - MCTS_4_INF_VANILLA",
    #                       "MEAN DEPTHS - MCTS-NC_1_INF_4_64_OCP_THRIFTY", "MAX DEPTHS - MCTS-NC_1_INF_4_64_OCP_THRIFTY",                                  
    #                       "MOVES ROUND", "MEAN DEPTHS, MAX_DEPTHS", None, "SAMPLE GAME OF CONNECT 4")

    # best_action_plot_generator("0500042733_17720_427_[mctsnc_30_inf_2_128_ocp_prodigal_16g;mctsnc_30_inf_2_128_acp_prodigal_16g;Gomoku_15x15;100]", 55,
    #                            "BEST $\widehat{q}$ - MCTS-NC_30_INF_2_128_OCP_PRODIGAL", "UCB - MCTS-NC_30_INF_2_128_OCP_PRODIGAL",
    #                            "BEST $\widehat{q}$ - MCTS-NC_30_INF_2_128_ACP_PRODIGAL", "UCB - MCTS-NC_30_INF_2_128_ACP_PRODIGAL",                                  
    #                            "MOVES ROUND", "MEAN DEPTHS, MAX_DEPTHS", None, "SAMPLE GAME OF GOMOKU")
    
    # depths_plot_generator("0500042733_17720_427_[mctsnc_30_inf_2_128_ocp_prodigal_16g;mctsnc_30_inf_2_128_acp_prodigal_16g;Gomoku_15x15;100]", 55,  
    #                       "MEAN DEPTHS - MCTS-NC_30_INF_2_128_OCP_PRODIGAL", "MAX DEPTHS - MCTS-NC_30_INF_2_128_OCP_PRODIGAL",
    #                       "MEAN DEPTHS - MCTS-NC_30_INF_2_128_ACP_PRODIGAL", "MAX DEPTHS - MCTS-NC_30_INF_2_128_ACP_PRODIGAL",                                  
    #                       "MOVES ROUND", "MEAN DEPTHS, MAX_DEPTHS", None, "SAMPLE GAME OF GOMOKU")
            
    # experiments_hs_array = np.array([
    #     ["1302514517_91136_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_32_ocp_prodigal;C4_6x7;100]",
    #      "2439094103_18658_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_64_ocp_prodigal;C4_6x7;100]",
    #      "0611358305_22348_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_128_ocp_prodigal;C4_6x7;100]",
    #      "3982047969_49772_427_[mcts_4_inf_vanilla;mctsnc_1_inf_1_256_ocp_prodigal;C4_6x7;100]"],
    #     ["1281441239_38466_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_32_ocp_prodigal;C4_6x7;100]",
    #      "2418020825_65988_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_64_ocp_prodigal;C4_6x7;100]",
    #      "2356598209_12844_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_128_ocp_prodigal;C4_6x7;100]",
    #      "1432320577_07564_427_[mcts_4_inf_vanilla;mctsnc_1_inf_2_256_ocp_prodigal;C4_6x7;100]"],
    #     ["1239294683_33126_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_32_ocp_prodigal;C4_6x7;100]",         
    #      "2375874269_93352_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_64_ocp_prodigal;C4_6x7;100]",
    #      "1552110721_61132_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_128_ocp_prodigal;C4_6x7;100]",
    #      "0627833089_55852_427_[mcts_4_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]"],
    #     ["1155001571_55150_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_32_ocp_prodigal;C4_6x7;100]",
    #      "2291581157_82672_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_64_ocp_prodigal;C4_6x7;100]",
    #      "4238103041_90412_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_128_ocp_prodigal;C4_6x7;100]",
    #      "3313825409_85132_427_[mcts_4_inf_vanilla;mctsnc_1_inf_8_256_ocp_prodigal;C4_6x7;100]"]
    #     ])
    # scores_curve_plot_generator(experiments_hs_array, "TIME LIMIT [s]", "AVERAGE SCORES", [1/4, 1/2, 2, 4], "MCTS-NC OCP-PRODIGAL (1s) \nvs MCTS VANILLA (4s)", "TITLE 2")