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
and that `c4` (instance of `C4(State)`) represents a state of an ongoing Connect 4 game shown below.
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

## Documentation
Complete developer documentation of the project is accessible at: [https://pklesk.github.io/mcts_numba_cuda](https://pklesk.github.io/mcts_numba_cuda). <br/>
Documentation for the `MCTSNC` class alone is at: [https://pklesk.github.io/mcts_numba_cuda/mctsnc.html](https://pklesk.github.io/mcts_numba_cuda/mctsnc.html).
