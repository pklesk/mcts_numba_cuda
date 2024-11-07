import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import numpy as np
from utils import unzip_and_load_experiment

FOLDER_EXPERIMENTS = "../experiments/"

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
        ["2325695625_84404_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_ocp_thrifty;C4_6x7;100]",
         "1183614633_99540_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_ocp_thrifty;C4_6x7;100]",
         "1599044513_98220_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_ocp_thrifty;C4_6x7;100]",
         "1046884313_08164_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_ocp_thrifty;C4_6x7;100]"],
        ["0494121001_62772_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_ocp_thrifty;C4_6x7;100]",
         "3647007305_45204_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_ocp_thrifty;C4_6x7;100]",
         "1577971235_45550_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_ocp_thrifty;C4_6x7;100]",
         "1025811035_55494_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_ocp_thrifty;C4_6x7;100]"],
        ["1125939049_86804_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_ocp_thrifty;C4_6x7;100]",
         "4278825353_01940_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_ocp_thrifty;C4_6x7;100]",
         "1535824679_72914_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_ocp_thrifty;C4_6x7;100]",
         "2163680100_50154_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_thrifty;C4_6x7;100]"],
        ["2389575145_67572_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_ocp_thrifty;C4_6x7;100]",
         "1247494153_50004_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_ocp_thrifty;C4_6x7;100]",
         "1451531567_62234_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_ocp_thrifty;C4_6x7;100]",
         "0899371367_72178_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_ocp_thrifty;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "OCP-THRIFTY (1$\,$s) vs VANILLA (5$\,$s)")    

def scores_array_plot_ocp_prodigal_vs_vanilla():    
    experiments_hs_array = np.array([
        ["2015967285_82784_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_ocp_prodigal;C4_6x7;100]",
         "3152546871_43010_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_ocp_prodigal;C4_6x7;100]",
         "3339668289_77068_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_ocp_prodigal;C4_6x7;100]",
         "2415390657_71788_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_ocp_prodigal;C4_6x7;100]"],
        ["1994894007_30114_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_ocp_prodigal;C4_6x7;100]",
         "3131473593_90340_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_ocp_prodigal;C4_6x7;100]",
         "0789940897_67564_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_ocp_prodigal;C4_6x7;100]",
         "4160630561_62284_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_ocp_prodigal;C4_6x7;100]"],
        ["3132763072_24774_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_ocp_prodigal;C4_6x7;100]",         
         "3089327037_85000_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_ocp_prodigal;C4_6x7;100]",
         "4280420705_15852_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_ocp_prodigal;C4_6x7;100]",
         "3356143073_10572_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]"],
        ["1868454339_46798_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_ocp_prodigal;C4_6x7;100]",
         "3005033925_07024_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_ocp_prodigal;C4_6x7;100]",
         "2671445729_45132_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_ocp_prodigal;C4_6x7;100]",
         "1747168097_39852_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_ocp_prodigal;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "OCP-PRODIGAL (1$\,$s) vs VANILLA (5$\,$s)")

def scores_array_plot_acp_thrifty_vs_vanilla():
    
    experiments_hs_array = np.array([
        ["3340130505_78932_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_acp_thrifty;C4_6x7;100]",
         "3378065134_94068_048_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_acp_thrifty;C4_6x7;100]",
         "4211774725_67120_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_acp_thrifty;C4_6x7;100]",         
         "3659614525_44360_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_acp_thrifty;C4_6x7;100]"],
        ["1508555881_57300_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_acp_thrifty;C4_6x7;100]",
         "0366474889_39732_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_acp_thrifty;C4_6x7;100]",
         "4190701447_14450_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_acp_thrifty;C4_6x7;100]",
         "3638541247_91690_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_acp_thrifty;C4_6x7;100]"],
        ["3320389550_81332_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_acp_thrifty;C4_6x7;100]",         
         "0998292937_96468_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_acp_thrifty;C4_6x7;100]",
         "4148554891_09110_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_acp_thrifty;C4_6x7;100]",
         "3596394691_19054_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_acp_thrifty;C4_6x7;100]"],
        ["3404010025_62100_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_acp_thrifty;C4_6x7;100]",
         "2261929033_44532_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_acp_thrifty;C4_6x7;100]",
         "4064261779_31134_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_acp_thrifty;C4_6x7;100]",
         "3512101579_08374_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_acp_thrifty;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "ACP-THRIFTY (1$\,$s) vs VANILLA (5$\,$s)")

def scores_array_plot_acp_prodigal_vs_vanilla():        
    experiments_hs_array = np.array([
        ["1406225233_62716_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_32_acp_prodigal;C4_6x7;100]",
         "2542804819_90238_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_64_acp_prodigal;C4_6x7;100]",
         "1093448321_77772_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_128_acp_prodigal;C4_6x7;100]",
         "0169170689_72492_427_[mcts_5_inf_vanilla;mctsnc_1_inf_1_256_acp_prodigal;C4_6x7;100]"],
        ["2565167576_10046_048_[mcts_5_inf_vanilla;mctsnc_1_inf_2_32_acp_prodigal;C4_6x7;100]",
         "2521731541_70272_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_64_acp_prodigal;C4_6x7;100]",
         "2838688225_35564_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_128_acp_prodigal;C4_6x7;100]",
         "1914410593_62988_427_[mcts_5_inf_vanilla;mctsnc_1_inf_2_256_acp_prodigal;C4_6x7;100]"],
        ["1343005399_04706_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_32_acp_prodigal;C4_6x7;100]",                  
         "3659600606_64932_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_64_acp_prodigal;C4_6x7;100]",
         "3214216358_16556_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_128_acp_prodigal;C4_6x7;100]",
         "2289938726_11276_048_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_acp_prodigal;C4_6x7;100]"],
        ["1258712287_26730_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_32_acp_prodigal;C4_6x7;100]",
         "2395291873_54252_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_64_acp_prodigal;C4_6x7;100]",
         "1605241382_13132_048_[mcts_5_inf_vanilla;mctsnc_1_inf_8_128_acp_prodigal;C4_6x7;100]",
         "3795915425_07852_427_[mcts_5_inf_vanilla;mctsnc_1_inf_8_256_acp_prodigal;C4_6x7;100]"]
        ])
    scores_array_plot_generator(experiments_hs_array, "$m$ (n_playouts)", "$T$ (n_trees)", [32, 64, 128, 256], [1, 2, 4, 8], "ACP-PRODIGAL (1$\,$s) vs VANILLA (5$\,$s)")

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
                     label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2,
                     ucbs_factor=1.0, ucbs_black_color=None, ucbs_white_color=None):
    figsize = (10.0, 5.0)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 11
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
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)    
    plt.grid(color=grid_color, zorder=0, dashes=grid_dashes)  
    plt.tight_layout(pad=0.4) 
    plt.show()
    
def best_action_plot_generator(experiments_hs, game_index, 
                               label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2,
                               ucbs_factor=1.0, ucbs_black_color=None, ucbs_white_color=None):
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
    best_action_plot(moves_rounds_black, qs_black, ucbs_black, moves_rounds_white, qs_white, ucbs_white, label_qs_black, label_ucbs_black, label_qs_white, label_ucbs_white, label_x, label_y, title_1, title_2, ucbs_factor, ucbs_black_color, ucbs_white_color)    

def depths_plot(moves_rounds_black, mean_depths_black, max_depths_black, moves_rounds_white, mean_depths_white, max_depths_white, 
                label_mean_depths_black, label_max_depths_black, label_mean_depths_white, label_max_depths_white, label_x, label_y, title_1, title_2):
    figsize = (10, 5.0)        
    fontsize_suptitle = 20
    fontsize_title = 21
    fontsize_ticks = 11
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

def averages_printout_generator(experiments_hs_array, ai_instance_name):
    print("MEANS PRINTOUT...")
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
    print(f"THE AVERAGES -> PLAYOUTS/STEPS: {np.mean(playouts)}/{np.mean(steps)}, MEAN DEPTH/MAX DEPTH: {np.mean(mean_depths)}/{np.mean(max_depths)}]")
    print("AVERAGES PRINTOUT GENERATOR DONE.")

def averages_printout_c4_5s_vanilla():    
    averages_printout_generator(np.array([
        "1779966119_01490_427_[mcts_5_inf_vanilla;mctsnc_5_inf_4_128_ocp_thrifty;C4_6x7;100]",
        "1569951977_89204_427_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "0725584456_47630_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "0503747630_89908_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTS(search_time_limit=5.0, search_steps_limit=inf, vanilla=True, ucb_c=2.0, seed: 0)")    
    
def averages_printout_c4_5s_ocp_thrifty():    
    averages_printout_generator(np.array([
        "1779966119_01490_427_[mcts_5_inf_vanilla;mctsnc_5_inf_4_128_ocp_thrifty;C4_6x7;100]",
        "1311471072_93670_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "3070453690_29088_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "3360218400_94374_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=128, variant='ocp_thrifty', device_memory=2.0, ucb_c=2.0, seed: 0)")  

def averages_printout_c4_5s_ocp_prodigal():    
    averages_printout_generator(np.array([
        "1569951977_89204_427_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "1311471072_93670_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "1714786244_73898_048_[mctsnc_5_inf_4_256_acp_thrifty;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "2504702716_35906_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='ocp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)")
    
def averages_printout_c4_5s_acp_thrifty():    
    averages_printout_generator(np.array([
        "0725584456_47630_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "3070453690_29088_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_thrifty;C4_6x7;100]",
        "1714786244_73898_048_[mctsnc_5_inf_4_256_acp_thrifty;mctsnc_5_inf_4_256_ocp_prodigal;C4_6x7;100]",
        "3763533572_41898_048_[mctsnc_5_inf_4_256_acp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_thrifty', device_memory=2.0, ucb_c=2.0, seed: 0)")    

def averages_printout_c4_5s_acp_prodigal():    
    averages_printout_generator(np.array([
        "0503747630_89908_048_[mcts_5_inf_vanilla;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]",
        "3360218400_94374_048_[mctsnc_5_inf_4_128_ocp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]",
        "2504702716_35906_048_[mctsnc_5_inf_4_256_ocp_prodigal;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]",
        "3763533572_41898_048_[mctsnc_5_inf_4_256_acp_thrifty;mctsnc_5_inf_4_256_acp_prodigal;C4_6x7;100]"
        ]), 
        "MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)")

def averages_printout_gomoku_30s_vanilla():
    averages_printout_generator(np.array([
        "3014955156_02650_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_128_ocp_thrifty_16g;Gomoku_15x15;100]",
        "1681612016_34230_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "4070724948_30746_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "3240654036_09850_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTS(search_time_limit=30.0, search_steps_limit=inf, vanilla=True, ucb_c=2.0, seed: 0)")


def averages_printout_gomoku_30s_ocp_thrifty():
    averages_printout_generator(np.array([
        "3014955156_02650_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_128_ocp_thrifty_16g;Gomoku_15x15;100]",
        "4039933000_53870_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "2602789548_96434_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "1304007724_29490_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=128, variant='ocp_thrifty', device_memory=16.0, ucb_c=2.0, seed: 0)")

def averages_printout_gomoku_30s_ocp_prodigal():
    averages_printout_generator(np.array([
        "1681612016_34230_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "4039933000_53870_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_ocp_prodigal_16g;Gomoku_15x15;100]",
        "1988707178_17328_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "3876337002_15280_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='ocp_prodigal', device_memory=16.0, ucb_c=2.0, seed: 0)")


def averages_printout_gomoku_30s_acp_thrifty():
    averages_printout_generator(np.array([
        "4070724948_30746_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "2602789548_96434_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "1988707178_17328_048_[mctsnc_30_inf_4_256_ocp_prodigal_16g;mctsnc_30_inf_4_256_acp_thrifty_16g;Gomoku_15x15;100]",
        "2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_thrifty', device_memory=16.0, ucb_c=2.0, seed: 0)")

def averages_printout_gomoku_30s_acp_prodigal():
    averages_printout_generator(np.array([
        "3240654036_09850_048_[mcts_30_inf_vanilla;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]",
        "1304007724_29490_048_[mctsnc_30_inf_4_128_ocp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]",
        "0500042733_17720_427_[mctsnc_30_inf_2_128_ocp_prodigal_16g;mctsnc_30_inf_2_128_acp_prodigal_16g;Gomoku_15x15;100]",
        "2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]"
        ]), 
        "MCTSNC(search_time_limit=30.0, search_steps_limit=inf, n_trees=4, n_playouts=256, variant='acp_prodigal', device_memory=16.0, ucb_c=2.0, seed: 0)")
    
if __name__ == "__main__":        
    
    scores_array_plot_ocp_thrifty_vs_vanilla()
    
    # scores_array_plot_ocp_prodigal_vs_vanilla()
    
    # scores_array_plot_acp_thrifty_vs_vanilla()
    
    # scores_array_plot_acp_prodigal_vs_vanilla()

    # best_action_plot_generator("3356143073_10572_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]", 61,
    #                            "BEST $\widehat{q}$ - MCTS_4_INF_VANILLA", "UCB - MCTS_4_INF_VANILLA",      
    #                            "BEST $\widehat{q}$ - MCTS-NC_1_INF_4_256_OCP_PRODIGAL", "UCB - MCTS-NC_1_INF_4_256_OCP_PRODIGAL",     
    #                            "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF CONNECT 4 (NO. 61/100)")
        
    # depths_plot_generator("3356143073_10572_427_[mcts_5_inf_vanilla;mctsnc_1_inf_4_256_ocp_prodigal;C4_6x7;100]", 61,  
    #                       "MEAN DEPTHS - MCTS_4_INF_VANILLA", "MAX DEPTHS - MCTS_4_INF_VANILLA",
    #                       "MEAN DEPTHS - MCTS-NC_1_INF_4_256_OCP_PRODIGAL", "MAX DEPTHS - MCTS-NC_1_INF_4_256_OCP_PRODIGAL",                                  
    #                       "MOVES ROUND", "MEAN, MAXIMUM DEPTHS", None, "SAMPLE GAME OF CONNECT 4 (NO. 61/100)")
    
    # best_action_plot_generator("2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]", 51,
    #                            "BEST $\widehat{q}$ - MCTS-NC_30_INF_4_256_ACP_THRIFTY", "25 x UCB - MCTS-NC_30_INF_4_256_ACP_THRIFTY",
    #                            "BEST $\widehat{q}$ - MCTS-NC_30_INF_4_256_ACP_PRODIGAL", "25 x UCB - MCTS-NC_30_INF_4_256_ACP_PRODIGAL",                                  
    #                            "MOVES ROUND", "BEST ACTIONS': $\widehat{q}$, UCB", None, "SAMPLE GAME OF GOMOKU (NO. 51/100)", 25.0)
    
    # depths_plot_generator("2094160108_21298_048_[mctsnc_30_inf_4_256_acp_thrifty_16g;mctsnc_30_inf_4_256_acp_prodigal_16g;Gomoku_15x15;100]", 51,  
    #                       "MEAN DEPTHS - MCTS-NC_30_INF_4_256_ACP_THRIFTY", "MAX DEPTHS - MCTS-NC_30_INF_4_256_ACP_THRIFTY",
    #                       "MEAN DEPTHS - MCTS-NC_30_INF_4_256_ACP_PRODIGAL", "MAX DEPTHS - MCTS-NC_30_INF_4_256_ACP_PRODIGAL",                                  
    #                       "MOVES ROUND", "MEAN, MAXIMUM DEPTHS", None, "SAMPLE GAME OF GOMOKU (NO. 51/100)")
        
    # averages_printout_c4_5s_vanilla()
    
    # averages_printout_c4_5s_ocp_thrifty()
    
    # averages_printout_c4_5s_ocp_prodigal()
    
    # averages_printout_c4_5s_acp_thrifty()
    
    # averages_printout_c4_5s_acp_prodigal()
    
    # averages_printout_gomoku_30s_vanilla()
    
    # averages_printout_gomoku_30s_ocp_thrifty()
    
    # averages_printout_gomoku_30s_ocp_prodigal()
    
    # averages_printout_gomoku_30s_acp_thrifty()
    
    # averages_printout_gomoku_30s_acp_prodigal()
            
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