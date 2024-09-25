UNDER CONSTRUCTION.

# MCTS-NC: A thorough GPU parallelization of Monte Carlo Tree Search implemented in Python via numba.cuda
With CUDA computational model in mind, we propose and implement four, fast operating and thoroughly parallel, variants of Monte Carlo Tree Search algorithm. 
The provided implementation takes advantage of [Numba](https://numba.pydata.org/), a just-in-time Python compiler, and its `numba.cuda` package (hence the "-NC" suffix in the project name). 
By *thoroughly parallel* we understand an algorithmic design that applies to both: (1) the structural elements of trees - leaf-/root-/tree-level parallelization 
(all those three are combined), and (2) the stages of MCTS - each stage in itself (selection, expansion, playouts, backup) employs multiple GPU threads. 
We apply suitable *reduction* patterns to carry out summations or max / argmax operations. Cooperation of threads helps to transfer information between global and shared memory. 
The implementation uses: no atomic operations, no mutexes (lock-free), and very few host-device memory transfers.

## High-level intuition 
<table>
   <tr><td><img src="https://github.com/user-attachments/assets/df115f08-a5a4-409d-8b93-de84be6133f2"/></td></tr>
</table>
<table>   
   <tr><td><img src="https://github.com/user-attachments/assets/fea4b1ec-25d2-459c-b519-3727ecd3268b"/></td></tr>
</table>
In MCTS-NC, there are two main variants according to which it conducts the playouts: OCP (*One Child Playouts*), ACP (*All Children Playouts*). 
Each of them has two subvariants, named "thrifty" and "prodigal".
In both OCP and ACP, multiple independent trees are grown concurrently (for readability just two are shown in each illustration).
Wavy arrows distinguished by colors represent CUDA threads working for different stages of MCTS algorithm:
orange for selection, green for expansion, black for playouts, purple for backup. In MCTS-NC, threads are grouped in 
CUDA blocks that are indexed either by tree indexes alone, or tree-action pairs, depending on the stage and variant / subvariant. 
In the OCP variant, exactly one random child of each expanded leaf node (accross different trees) becomes played out. 
In ACP, all such children become played out. In the figure, terminal rewards from playouts are colored
in: blue (losses of the first "red" player), gray (draws) or red (wins of the first player). Their counts suitably update
the statistics at ancestor nodes. For shortness, Q stands for an action-value estimate and U for its upper confidence bound.

## Example usage 1 (Connect 4)
Assume the specifics of the Connect 4 game have been defined to MCTS-NC in `mctsnc_game_specifics.py` module (i.e. functions `is_action_legal`, `take_action`, etc.), 
and that `c4` - instance of `C4(State)` - represents a state of an ongoing Connect 4 game shown below.
```bash
|.|.|●|○|.|.|.|
|.|.|●|○|.|.|○|
|.|.|●|●|.|●|●|
|.|●|○|●|.|○|●|
|.|○|●|○|.|●|○|
|○|○|○|●|●|○|○|
 0 1 2 3 4 5 6 
```
Then, running the following code
```python
ai = MCTSNC(C4.get_board_shape(), C4.get_extra_info_memory(), C4.get_max_actions())
ai.init_device_side_arrays()
best_action = ai.run(c4.get_board(), c4.get_extra_info(), c4.get_turn())
print(f"BEST ACTION: {best_action}")
```
results in finding the best action for black - move 4 (winning in two plies), and the following printout:
```bash
[MCTSNC._init_device_side_arrays()... for MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
[MCTSNC._init_device_side_arrays() done; time: 0.5193691253662109 s, per_state_memory: 95 B,  calculated max_tree_size: 2825549]
MCTSNC RUN... [MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
[actions info:
{
  0: {'name': '0', 'n_root': 7474304, 'win_flag': False, 'n': 2182400, 'n_wins': 2100454, 'q': 0.9624514296187683, 'ucb': 0.9678373740384631},
  1: {'name': '1', 'n_root': 7474304, 'win_flag': False, 'n': 185344, 'n_wins': 164757, 'q': 0.8889254575276243, 'ucb': 0.9074070665330406},
  4: {'name': '4', 'n_root': 7474304, 'win_flag': False, 'n': 4921472, 'n_wins': 4885924, 'q': 0.9927769577882389, 'ucb': 0.9963635461474457},
  5: {'name': '5', 'n_root': 7474304, 'win_flag': False, 'n': 105472, 'n_wins': 91863, 'q': 0.8709704945388349, 'ucb': 0.8954701768685893},
  6: {'name': '6', 'n_root': 7474304, 'win_flag': False, 'n': 79616, 'n_wins': 68403, 'q': 0.8591614750803859, 'ucb': 0.8873601607647162},
  best: {'index': 4, 'name': '4', 'n_root': 7474304, 'win_flag': False, 'n': 4921472, 'n_wins': 4885924, 'q': 0.9927769577882389, 'ucb': 0.9963635461474457}
}]
[performance info:
{
  steps: 6373,
  steps_per_second: 1274.0076324260813,
  playouts: 7474304,
  playouts_per_second: 1494166.0666990099,
  times_[ms]: {'total': 5002.324819564819, 'loop': 5000.642776489258, 'reduce_over_trees': 0.29015541076660156, 'reduce_over_actions': 0.4520416259765625, 'mean_loop': 0.7846607212441955, 'mean_select': 0.11222893376562147, 'mean_expand': 0.2786097114284054, 'mean_playout': 0.17186361935680036, 'mean_backup': 0.2193056618645448},
  trees: {'count': 8, 'mean_depth': 5.176703163017032, 'max_depth': 12, 'mean_size': 1233.0, 'max_size': 2736}
}]
MCTSNC RUN DONE. [time: 5.002324819564819 s; best action: 4, best win_flag: False, best n: 4921472, best n_wins: 4885924, best q: 0.9927769577882389]
BEST ACTION: 4
```

## Example usage 2 (Gomoku)
Assume the specifics of the Gomoku game have been defined to MCTS-NC in `mctsnc_game_specifics.py` module (i.e. functions `is_action_legal`, `take_action`, etc.), 
and that `g` - instance of `Gomoku(State)` - represents a state of an ongoing Gomoku game shown below.
```bash
  ABCDEFGHIJKLMNO
15+++++++++++++++15
14+++++++++++++++14
13+++++++++++++++13
12++++++++●++++++12
11++++++++○++++++11
10++++++++○++++++10
 9++++++○+○++++++9
 8+++++++●○++++++8
 7+++++++●●●○++++7
 6++++++++●●○++++6
 5+++++++●+++++++5
 4+++++++++++++++4
 3+++++++++++++++3
 2+++++++++++++++2
 1+++++++++++++++1
  ABCDEFGHIJKLMNO
Then, running the following code
```python
ai = MCTSNC(Gomoku.get_board_shape(), Gomoku.get_extra_info_memory(), Gomoku.get_max_actions(), action_index_to_name_function=Gomoku.action_index_to_name)
ai.init_device_side_arrays()
best_action = ai.run(g.get_board(), g.get_extra_info(), g.get_turn())
print(f"BEST ACTION: {best_action}")
```
results in finding the best action for white - move K8 (indexed as 115), preventing black from winning in two plies, and the following printout:
<div style="max-height: 20px; overflow-y: auto;">
<pre><code>
[MCTSNC._init_device_side_arrays()... for MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
[MCTSNC._init_device_side_arrays() done; time: 0.5558419227600098 s, per_state_memory: 1144 B,  calculated max_tree_size: 234637]
MCTSNC RUN... [MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
[actions info:
{
  0: {'name': 'A1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148906, 'q': 0.3478852048444976, 'ucb': 0.36098484108863044},
  1: {'name': 'B1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149000, 'q': 0.34810481459330145, 'ucb': 0.3612044508374343},
  2: {'name': 'C1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 144339, 'q': 0.3372154418361244, 'ucb': 0.35031507808025725},
  3: {'name': 'D1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152655, 'q': 0.3566438957834928, 'ucb': 0.36974353202762567},
  4: {'name': 'E1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151385, 'q': 0.3536768279007177, 'ucb': 0.36677646414485054},
  5: {'name': 'F1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147897, 'q': 0.3455279044557416, 'ucb': 0.35862754069987446},
  6: {'name': 'G1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153681, 'q': 0.35904091282894735, 'ucb': 0.3721405490730802},
  7: {'name': 'H1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147007, 'q': 0.3434486206638756, 'ucb': 0.35654825690800845},
  8: {'name': 'I1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 154665, 'q': 0.3613398063696172, 'ucb': 0.37443944261375006},
  9: {'name': 'J1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 146122, 'q': 0.3413810182416268, 'ucb': 0.35448065448575966},
  10: {'name': 'K1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151115, 'q': 0.35304603394138756, 'ucb': 0.3661456701855204},
  11: {'name': 'L1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147381, 'q': 0.344322387111244, 'ucb': 0.35742202335537687},
  12: {'name': 'M1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150016, 'q': 0.35047846889952156, 'ucb': 0.3635781051436544},
  13: {'name': 'N1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147014, 'q': 0.34346497458133973, 'ucb': 0.3565646108254726},
  14: {'name': 'O1', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 138825, 'q': 0.34584512515943877, 'ucb': 0.35937221403831915},
  15: {'name': 'A2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 145043, 'q': 0.33886017867822965, 'ucb': 0.3519598149223625},
  16: {'name': 'B2', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 139366, 'q': 0.34719288105867346, 'ucb': 0.36071996993755384},
  17: {'name': 'C2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147797, 'q': 0.3452942770633971, 'ucb': 0.35839391330752995},
  18: {'name': 'D2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150048, 'q': 0.3505532296650718, 'ucb': 0.36365286590920465},
  19: {'name': 'E2', 'n_root': 94359552, 'win_flag': False, 'n': 374784, 'n_wins': 129259, 'q': 0.3448893229166667, 'ucb': 0.3588886394833727},
  20: {'name': 'F2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148435, 'q': 0.346784819826555, 'ucb': 0.35988445607068786},
  21: {'name': 'G2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148551, 'q': 0.34705582760167464, 'ucb': 0.3601554638458075},
  22: {'name': 'H2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147750, 'q': 0.34518447218899523, 'ucb': 0.3582841084331281},
  23: {'name': 'I2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 145924, 'q': 0.34091843600478466, 'ucb': 0.3540180722489175},
  24: {'name': 'J2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149975, 'q': 0.35038268166866027, 'ucb': 0.3634823179127931},
  25: {'name': 'K2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147516, 'q': 0.3446377840909091, 'ucb': 0.35773742033504197},
  26: {'name': 'L2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150603, 'q': 0.35184986169258375, 'ucb': 0.3649494979367166},
  27: {'name': 'M2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148179, 'q': 0.34618673370215314, 'ucb': 0.359286369946286},
  28: {'name': 'N2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 144591, 'q': 0.33780418286483255, 'ucb': 0.3509038191089654},
  29: {'name': 'O2', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 146168, 'q': 0.34148848684210525, 'ucb': 0.3545881230862381},
  30: {'name': 'A3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149707, 'q': 0.34975656025717705, 'ucb': 0.3628561965013099},
  31: {'name': 'B3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151248, 'q': 0.35335675837320574, 'ucb': 0.3664563946173386},
  32: {'name': 'C3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 155822, 'q': 0.36404287529904306, 'ucb': 0.3771425115431759},
  33: {'name': 'D3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150134, 'q': 0.35075414922248804, 'ucb': 0.3638537854666209},
  34: {'name': 'E3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152618, 'q': 0.35655745364832536, 'ucb': 0.3696570898924582},
  35: {'name': 'F3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153678, 'q': 0.35903390400717705, 'ucb': 0.3721335402513099},
  36: {'name': 'G3', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 171655, 'q': 0.3566634807180851, 'ucb': 0.36901722046942087},
  37: {'name': 'H3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 146330, 'q': 0.3418669632177033, 'ucb': 0.3549665994618362},
  38: {'name': 'I3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152985, 'q': 0.35741486617822965, 'ucb': 0.3705145024223625},
  39: {'name': 'J3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148151, 'q': 0.3461213180322967, 'ucb': 0.3592209542764295},
  40: {'name': 'K3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 154268, 'q': 0.36041230562200954, 'ucb': 0.3735119418661424},
  41: {'name': 'L3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153085, 'q': 0.35764849357057416, 'ucb': 0.370748129814707},
  42: {'name': 'M3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151138, 'q': 0.3530997682416268, 'ucb': 0.36619940448575966},
  43: {'name': 'N3', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162833, 'q': 0.35814549901463966, 'ucb': 0.37085580166931004},
  44: {'name': 'O3', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150462, 'q': 0.351520447069378, 'ucb': 0.36462008331351087},
  45: {'name': 'A4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149826, 'q': 0.35003457685406697, 'ucb': 0.3631342130981998},
  46: {'name': 'B4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149270, 'q': 0.3487356085526316, 'ucb': 0.36183524479676443},
  47: {'name': 'C4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147597, 'q': 0.34482702227870815, 'ucb': 0.357926658522841},
  48: {'name': 'D4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152823, 'q': 0.3570363898026316, 'ucb': 0.37013602604676443},
  49: {'name': 'E4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149052, 'q': 0.3482263008373206, 'ucb': 0.36132593708145344},
  50: {'name': 'F4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 157950, 'q': 0.369014466208134, 'ucb': 0.38211410245226685},
  51: {'name': 'G4', 'n_root': 94359552, 'win_flag': False, 'n': 587776, 'n_wins': 227612, 'q': 0.3872427591463415, 'ucb': 0.39842146248211613},
  52: {'name': 'H4', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 182767, 'q': 0.37975191156914895, 'ucb': 0.3921056513204847},
  53: {'name': 'I4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 158956, 'q': 0.3713647577751196, 'ucb': 0.3844643940192525},
  54: {'name': 'J4', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 160060, 'q': 0.35204638231981983, 'ucb': 0.3647566849744902},
  55: {'name': 'K4', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 171727, 'q': 0.3777075415259009, 'ucb': 0.3904178441805713},
  56: {'name': 'L4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 158764, 'q': 0.3709161931818182, 'ucb': 0.384015829425951},
  57: {'name': 'M4', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 164114, 'q': 0.36096301379504503, 'ucb': 0.3736733164497154},
  58: {'name': 'N4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150840, 'q': 0.3524035586124402, 'ucb': 0.36550319485657307},
  59: {'name': 'O4', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151528, 'q': 0.35401091507177035, 'ucb': 0.3671105513159032},
  60: {'name': 'A5', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153925, 'q': 0.35961096366626794, 'ucb': 0.3727105999104008},
  61: {'name': 'B5', 'n_root': 94359552, 'win_flag': False, 'n': 507904, 'n_wins': 184913, 'q': 0.3640707692792339, 'ucb': 0.3760963633181586},
  62: {'name': 'C5', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149568, 'q': 0.3494318181818182, 'ucb': 0.362531454425951},
  63: {'name': 'D5', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162044, 'q': 0.35641012105855857, 'ucb': 0.36912042371322895},
  64: {'name': 'E5', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 163394, 'q': 0.35937939893018017, 'ucb': 0.37208970158485055},
  65: {'name': 'F5', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 164474, 'q': 0.3617548212274775, 'ucb': 0.37446512388214787},
  66: {'name': 'G5', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 165502, 'q': 0.3640158713400901, 'ucb': 0.3767261739947605},
  68: {'name': 'I5', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162530, 'q': 0.35747906109234234, 'ucb': 0.3701893637470127},
  69: {'name': 'J5', 'n_root': 94359552, 'win_flag': False, 'n': 507904, 'n_wins': 190847, 'q': 0.3757540795110887, 'ucb': 0.3877796735500134},
  70: {'name': 'K5', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 166864, 'q': 0.38984001196172247, 'ucb': 0.4029396482058553},
  71: {'name': 'L5', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 172593, 'q': 0.3586124501329787, 'ucb': 0.37096618988431446},
  72: {'name': 'M5', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 155403, 'q': 0.3630639765251196, 'ucb': 0.3761636127692525},
  73: {'name': 'N5', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149913, 'q': 0.3502378326854067, 'ucb': 0.36333746892953955},
  74: {'name': 'O5', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153401, 'q': 0.3583867561303828, 'ucb': 0.37148639237451564},
  75: {'name': 'A6', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 156554, 'q': 0.36575302781100477, 'ucb': 0.3788526640551376},
  76: {'name': 'B6', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 146977, 'q': 0.3433785324461722, 'ucb': 0.3564781686903051},
  77: {'name': 'C6', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148348, 'q': 0.34658156399521534, 'ucb': 0.3596812002393482},
  78: {'name': 'D6', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 176604, 'q': 0.3669464760638298, 'ucb': 0.37930021581516554},
  79: {'name': 'E6', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 161625, 'q': 0.3554885451858108, 'ucb': 0.3681988478404812},
  80: {'name': 'F6', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152463, 'q': 0.35619533119019137, 'ucb': 0.3692949674343242},
  81: {'name': 'G6', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 163646, 'q': 0.35993366413288286, 'ucb': 0.37264396678755324},
  82: {'name': 'H6', 'n_root': 94359552, 'win_flag': False, 'n': 587776, 'n_wins': 231550, 'q': 0.39394259037456447, 'ucb': 0.4051212937103391},
  86: {'name': 'L6', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 162710, 'q': 0.38013513008373206, 'ucb': 0.3932347663278649},
  87: {'name': 'M6', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 167245, 'q': 0.36784953899211714, 'ucb': 0.3805598416467875},
  88: {'name': 'N6', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162035, 'q': 0.35639032587274777, 'ucb': 0.36910062852741815},
  89: {'name': 'O6', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152969, 'q': 0.35737748579545453, 'ucb': 0.3704771220395874},
  90: {'name': 'A7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 146726, 'q': 0.34279212769138756, 'ucb': 0.3558917639355204},
  91: {'name': 'B7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152236, 'q': 0.3556649970095694, 'ucb': 0.36876463325370223},
  92: {'name': 'C7', 'n_root': 94359552, 'win_flag': False, 'n': 534528, 'n_wins': 198189, 'q': 0.37077384159482757, 'ucb': 0.3824961225330963},
  93: {'name': 'D7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150570, 'q': 0.35177276465311, 'ucb': 0.3648724008972429},
  94: {'name': 'E7', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 165270, 'q': 0.3635055954391892, 'ucb': 0.3762158980938596},
  95: {'name': 'F7', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 166814, 'q': 0.3669015695382883, 'ucb': 0.37961187219295867},
  96: {'name': 'G7', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 169234, 'q': 0.3722242750563063, 'ucb': 0.38493457771097667},
  101: {'name': 'L7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 155132, 'q': 0.362430846291866, 'ucb': 0.37553048253599886},
  102: {'name': 'M7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147893, 'q': 0.34551855936004783, 'ucb': 0.3586181956041807},
  103: {'name': 'N7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150076, 'q': 0.3506186453349282, 'ucb': 0.36371828157906105},
  104: {'name': 'O7', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150300, 'q': 0.3511419706937799, 'ucb': 0.36424160693791274},
  105: {'name': 'A8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149868, 'q': 0.3501327003588517, 'ucb': 0.36323233660298454},
  106: {'name': 'B8', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 168118, 'q': 0.3493143284574468, 'ucb': 0.36166806820878256},
  107: {'name': 'C8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149911, 'q': 0.3502331601375598, 'ucb': 0.36333279638169264},
  108: {'name': 'D8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149621, 'q': 0.34955564069976075, 'ucb': 0.3626552769438936},
  109: {'name': 'E8', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 166363, 'q': 0.36590961078265766, 'ucb': 0.37861991343732804},
  110: {'name': 'F8', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 170418, 'q': 0.37482844172297297, 'ucb': 0.38753874437764335},
  111: {'name': 'G8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 158858, 'q': 0.371135802930622, 'ucb': 0.38423543917475483},
  114: {'name': 'J8', 'n_root': 94359552, 'win_flag': False, 'n': 641024, 'n_wins': 254466, 'q': 0.39696797623801916, 'ucb': 0.40767232401153985},
  115: {'name': 'K8', 'n_root': 94359552, 'win_flag': False, 'n': 1093632, 'n_wins': 452284, 'q': 0.41356141736891383, 'ucb': 0.4217566587685248},
  116: {'name': 'L8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151405, 'q': 0.3537235533791866, 'ucb': 0.36682318962331945},
  117: {'name': 'M8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153316, 'q': 0.35818817284689, 'ucb': 0.3712878090910228},
  118: {'name': 'N8', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 163375, 'q': 0.35933760909346846, 'ucb': 0.37204791174813884},
  119: {'name': 'O8', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152354, 'q': 0.35594067733253587, 'ucb': 0.3690403135766687},
  120: {'name': 'A9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151812, 'q': 0.3546744168660287, 'ucb': 0.36777405311016154},
  121: {'name': 'B9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148515, 'q': 0.3469717217404306, 'ucb': 0.36007135798456347},
  122: {'name': 'C9', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 161825, 'q': 0.35592843820382886, 'ucb': 0.36863874085849924},
  123: {'name': 'D9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 154151, 'q': 0.3601389615729665, 'ucb': 0.37323859781709934},
  124: {'name': 'E9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 155599, 'q': 0.36352188621411485, 'ucb': 0.3766215224582477},
  125: {'name': 'F9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 161942, 'q': 0.3783408717105263, 'ucb': 0.3914405079546592},
  127: {'name': 'H9', 'n_root': 94359552, 'win_flag': False, 'n': 1013760, 'n_wins': 419112, 'q': 0.4134232954545455, 'ucb': 0.42193525948543675},
  129: {'name': 'J9', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 184130, 'q': 0.3825839428191489, 'ucb': 0.39493768257048467},
  130: {'name': 'K9', 'n_root': 94359552, 'win_flag': False, 'n': 694272, 'n_wins': 273925, 'q': 0.39454997464970504, 'ucb': 0.40483564331353894},
  131: {'name': 'L9', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 161080, 'q': 0.3542898367117117, 'ucb': 0.3670001393663821},
  132: {'name': 'M9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 157054, 'q': 0.3669211647727273, 'ucb': 0.38002080101686014},
  133: {'name': 'N9', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 161598, 'q': 0.3554291596283784, 'ucb': 0.3681394622830488},
  134: {'name': 'O9', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152744, 'q': 0.3568518241626794, 'ucb': 0.36995146040681226},
  135: {'name': 'A10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147078, 'q': 0.3436144961124402, 'ucb': 0.35671413235657307},
  136: {'name': 'B10', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 172418, 'q': 0.3582488364361702, 'ucb': 0.37060257618750597},
  137: {'name': 'C10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151167, 'q': 0.3531675201854067, 'ucb': 0.36626715642953955},
  138: {'name': 'D10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153206, 'q': 0.357931182715311, 'ucb': 0.37103081895944384},
  139: {'name': 'E10', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 176727, 'q': 0.36720204454787236, 'ucb': 0.3795557842992081},
  140: {'name': 'F10', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 178971, 'q': 0.37186461103723406, 'ucb': 0.3842183507885698},
  141: {'name': 'G10', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 180292, 'q': 0.374609375, 'ucb': 0.38696311475133577},
  142: {'name': 'H10', 'n_root': 94359552, 'win_flag': False, 'n': 827392, 'n_wins': 340611, 'q': 0.4116682298112624, 'ucb': 0.4210901993680957},
  144: {'name': 'J10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 155677, 'q': 0.36370411558014354, 'ucb': 0.3768037518242764},
  145: {'name': 'K10', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 164793, 'q': 0.36245645059121623, 'ucb': 0.3751667532458866},
  146: {'name': 'L10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149948, 'q': 0.3503196022727273, 'ucb': 0.36341923851686014},
  147: {'name': 'M10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153053, 'q': 0.3575737328050239, 'ucb': 0.37067336904915676},
  148: {'name': 'N10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149069, 'q': 0.34826601749401914, 'ucb': 0.361365653738152},
  149: {'name': 'O10', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150601, 'q': 0.35184518914473684, 'ucb': 0.3649448253888697},
  150: {'name': 'A11', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 158284, 'q': 0.34814013231981983, 'ucb': 0.3608504349744902},
  151: {'name': 'B11', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 161175, 'q': 0.3544987858952703, 'ucb': 0.36720908854994067},
  152: {'name': 'C11', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162041, 'q': 0.3564035226632883, 'ucb': 0.36911382531795867},
  153: {'name': 'D11', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 175605, 'q': 0.3648707613031915, 'ucb': 0.37722450105452726},
  154: {'name': 'E11', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 166470, 'q': 0.3661449535472973, 'ucb': 0.3788552562019677},
  155: {'name': 'F11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152652, 'q': 0.35663688696172247, 'ucb': 0.3697365232058553},
  156: {'name': 'G11', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 177637, 'q': 0.3690928357712766, 'ucb': 0.38144657552261235},
  157: {'name': 'H11', 'n_root': 94359552, 'win_flag': False, 'n': 641024, 'n_wins': 243836, 'q': 0.3803851337859425, 'ucb': 0.3910894815594632},
  159: {'name': 'J11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 158914, 'q': 0.3712666342703349, 'ucb': 0.38436627051446776},
  160: {'name': 'K11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 154566, 'q': 0.3611085152511962, 'ucb': 0.37420815149532904},
  161: {'name': 'L11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150928, 'q': 0.3526091507177033, 'ucb': 0.3657087869618362},
  162: {'name': 'M11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149224, 'q': 0.34862813995215314, 'ucb': 0.361727776196286},
  163: {'name': 'N11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148894, 'q': 0.34785716955741625, 'ucb': 0.3609568058015491},
  164: {'name': 'O11', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151012, 'q': 0.3528053977272727, 'ucb': 0.36590503397140556},
  165: {'name': 'A12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 154100, 'q': 0.36001981160287083, 'ucb': 0.3731194478470037},
  166: {'name': 'B12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 145963, 'q': 0.34100955068779903, 'ucb': 0.3541091869319319},
  167: {'name': 'C12', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162305, 'q': 0.35698418144707206, 'ucb': 0.36969448410174244},
  168: {'name': 'D12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148858, 'q': 0.3477730636961722, 'ucb': 0.3608726999403051},
  169: {'name': 'E12', 'n_root': 94359552, 'win_flag': False, 'n': 507904, 'n_wins': 188759, 'q': 0.37164306640625, 'ucb': 0.3836686604451747},
  170: {'name': 'F12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153062, 'q': 0.3575947592703349, 'ucb': 0.37069439551446776},
  171: {'name': 'G12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153363, 'q': 0.35829797772129185, 'ucb': 0.3713976139654247},
  172: {'name': 'H12', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 164607, 'q': 0.3620473500844595, 'ucb': 0.37475765273912987},
  174: {'name': 'J12', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 171543, 'q': 0.37730283994932434, 'ucb': 0.3900131426039947},
  175: {'name': 'K12', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 176798, 'q': 0.36734956781914896, 'ucb': 0.3797033075704847},
  176: {'name': 'L12', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 180360, 'q': 0.374750664893617, 'ucb': 0.38710440464495277},
  177: {'name': 'M12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148624, 'q': 0.3472263755980861, 'ucb': 0.36032601184221896},
  178: {'name': 'N12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 157051, 'q': 0.36691415595095694, 'ucb': 0.3800137921950898},
  179: {'name': 'O12', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149889, 'q': 0.350181762111244, 'ucb': 0.36328139835537687},
  180: {'name': 'A13', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 139868, 'q': 0.3484434789540816, 'ucb': 0.361970567832962},
  181: {'name': 'B13', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 138985, 'q': 0.3462437220982143, 'ucb': 0.3597708109770947},
  182: {'name': 'C13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151068, 'q': 0.35293622906698563, 'ucb': 0.3660358653111185},
  183: {'name': 'D13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153810, 'q': 0.3593422921650718, 'ucb': 0.37244192840920465},
  184: {'name': 'E13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152399, 'q': 0.3560458096590909, 'ucb': 0.36914544590322373},
  185: {'name': 'F13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149857, 'q': 0.3501070013456938, 'ucb': 0.36320663758982663},
  186: {'name': 'G13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 152492, 'q': 0.3562630831339713, 'ucb': 0.36936271937810417},
  187: {'name': 'H13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 151287, 'q': 0.3534478730562201, 'ucb': 0.36654750930035296},
  188: {'name': 'I13', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 166390, 'q': 0.3659689963400901, 'ucb': 0.3786792989947605},
  189: {'name': 'J13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 153528, 'q': 0.35868346291866027, 'ucb': 0.3717830991627931},
  190: {'name': 'K13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 156025, 'q': 0.3645171389055024, 'ucb': 0.37761677514963526},
  191: {'name': 'L13', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 174076, 'q': 0.3616938164893617, 'ucb': 0.37404755624069747},
  192: {'name': 'M13', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 165266, 'q': 0.36349679757882886, 'ucb': 0.37620710023349924},
  193: {'name': 'N13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150459, 'q': 0.35151343824760767, 'ucb': 0.3646130744917405},
  194: {'name': 'O13', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148372, 'q': 0.346637634569378, 'ucb': 0.35973727081351087},
  195: {'name': 'A14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 144453, 'q': 0.3374817770633971, 'ucb': 0.35058141330752995},
  196: {'name': 'B14', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 140538, 'q': 0.3501126036352041, 'ucb': 0.36363969251408446},
  197: {'name': 'C14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150596, 'q': 0.3518335077751196, 'ucb': 0.3649331440192525},
  198: {'name': 'D14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149823, 'q': 0.3500275680322967, 'ucb': 0.3631272042764295},
  199: {'name': 'E14', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 164873, 'q': 0.36263240779842343, 'ucb': 0.3753427104530938},
  200: {'name': 'F14', 'n_root': 94359552, 'win_flag': False, 'n': 481280, 'n_wins': 172993, 'q': 0.3594435671542553, 'ucb': 0.37179730690559104},
  201: {'name': 'G14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147132, 'q': 0.3437406549043062, 'ucb': 0.35684029114843907},
  202: {'name': 'H14', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 162243, 'q': 0.3568478146114865, 'ucb': 0.3695581172661569},
  203: {'name': 'I14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149636, 'q': 0.34959068480861244, 'ucb': 0.3626903210527453},
  204: {'name': 'J14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148415, 'q': 0.3467380943480861, 'ucb': 0.35983773059221896},
  205: {'name': 'K14', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 157952, 'q': 0.3474099099099099, 'ucb': 0.36012021256458027},
  206: {'name': 'L14', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149926, 'q': 0.3502682042464115, 'ucb': 0.3633678404905443},
  207: {'name': 'M14', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 140525, 'q': 0.35008021763392855, 'ucb': 0.3636073065128089},
  208: {'name': 'N14', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 140141, 'q': 0.3491235849808674, 'ucb': 0.36265067385974775},
  209: {'name': 'O14', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 140598, 'q': 0.3502620774872449, 'ucb': 0.3637891663661253},
  210: {'name': 'A15', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 160941, 'q': 0.3539841110641892, 'ucb': 0.3666944137188596},
  211: {'name': 'B15', 'n_root': 94359552, 'win_flag': False, 'n': 374784, 'n_wins': 129530, 'q': 0.345612406079235, 'ucb': 0.359611722645941},
  212: {'name': 'C15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147942, 'q': 0.3456330367822967, 'ucb': 0.3587326730264295},
  213: {'name': 'D15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148318, 'q': 0.34651147577751196, 'ucb': 0.3596111120216448},
  214: {'name': 'E15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150883, 'q': 0.3525040183911483, 'ucb': 0.36560365463528116},
  215: {'name': 'F15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148607, 'q': 0.34718665894138756, 'ucb': 0.3602862951855204},
  216: {'name': 'G15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 154838, 'q': 0.3617439817583732, 'ucb': 0.37484361800250604},
  217: {'name': 'H15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 147772, 'q': 0.345235870215311, 'ucb': 0.35833550645944384},
  218: {'name': 'I15', 'n_root': 94359552, 'win_flag': False, 'n': 454656, 'n_wins': 159105, 'q': 0.34994589315878377, 'ucb': 0.36265619581345415},
  219: {'name': 'J15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 150794, 'q': 0.3522960900119617, 'ucb': 0.36539572625609457},
  220: {'name': 'K15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149230, 'q': 0.3486421575956938, 'ucb': 0.36174179383982663},
  221: {'name': 'L15', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 141190, 'q': 0.3517368861607143, 'ucb': 0.3652639750395947},
  222: {'name': 'M15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148009, 'q': 0.34578956713516745, 'ucb': 0.3588892033793003},
  223: {'name': 'N15', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 148802, 'q': 0.37070013552295916, 'ucb': 0.38422722440183954},
  224: {'name': 'O15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 145329, 'q': 0.3395283530203349, 'ucb': 0.35262798926446776},
  best: {'index': 115, 'name': 'K8', 'n_root': 94359552, 'win_flag': False, 'n': 1093632, 'n_wins': 452284, 'q': 0.41356141736891383, 'ucb': 0.4217566587685248}
}]
[performance info:
{
  steps: 442,
  steps_per_second: 88.25552729358726,
  playouts: 94359552,
  playouts_per_second: 18841067.91164404,
  times_[ms]: {'total': 5008.184909820557, 'loop': 5006.503105163574, 'reduce_over_trees': 0.20575523376464844, 'reduce_over_actions': 0.5161762237548828, 'mean_loop': 11.326930102180032, 'mean_select': 0.10066766005295974, 'mean_expand': 0.3082833139065704, 'mean_playout': 10.688265524298897, 'mean_backup': 0.226746317488036},
  trees: {'count': 8, 'mean_depth': 2.519115779878241, 'max_depth': 3, 'mean_size': 92149.0, 'max_size': 92149}
}]
MCTSNC RUN DONE. [time: 5.008184909820557 s; best action: 115 (K8), best win_flag: False, best n: 1093632, best n_wins: 452284, best q: 0.41356141736891383]
BEST ACTION: 115
</code></pre>
</div>

## Documentation
Complete developer documentation of the project is accessible at: [https://pklesk.github.io/mcts_numba_cuda](https://pklesk.github.io/mcts_numba_cuda). <br/>
Documentation for the `MCTSNC` class alone is at: [https://pklesk.github.io/mcts_numba_cuda/mctsnc.html](https://pklesk.github.io/mcts_numba_cuda/mctsnc.html).
