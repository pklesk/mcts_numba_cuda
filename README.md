UNDER CONSTRUCTION.

## Example usage 1 (Connect 4)
Assuming `c4` represents a state of Connect 4 game - an instance of `C4(State)` class - shown below:
```bash
|.|.|●|○|.|.|.|
|.|.|●|○|.|.|○|
|.|.|●|●|.|●|●|
|.|●|○|●|.|○|●|
|.|○|●|○|.|●|○|
|○|○|○|●|●|○|○|
 0 1 2 3 4 5 6 
```
running the code
```python
ai = MCTSNC(C4.get_board_shape(), c4.get_extra_info_memory(), c4.get_max_actions())
ai.init_device_side_arrays()
best_action = ai.run(c4.get_board(), c4.get_extra_info(), c4.turn)
```
results in the following printout and finds the best action - move 4 - for black (to move now):
```bash
[MCTSNC._init_device_side_arrays()... for MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
[MCTSNC._init_device_side_arrays() done; time: 0.7248117923736572 s, per_state_memory: 95 B,  calculated max_tree_size: 2825549]
MCTSNC RUN... [MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
[actions info:
{
  0: {'name': '0', 'n_root': 3195776, 'win_flag': False, 'n': 1070464, 'n_wins': 996419, 'q': 0.930829061042688, 'ucb': 0.9383100896251532},
  1: {'name': '1', 'n_root': 3195776, 'win_flag': False, 'n': 185344, 'n_wins': 164757, 'q': 0.8889254575276243, 'ucb': 0.9069041490759143},
  4: {'name': '4', 'n_root': 3195776, 'win_flag': False, 'n': 1754880, 'n_wins': 1721240, 'q': 0.980830598103574, 'ucb': 0.9866734332771923},
  5: {'name': '5', 'n_root': 3195776, 'win_flag': False, 'n': 105472, 'n_wins': 91863, 'q': 0.8709704945388349, 'ucb': 0.8948034969310324},
  6: {'name': '6', 'n_root': 3195776, 'win_flag': False, 'n': 79616, 'n_wins': 68403, 'q': 0.8591614750803859, 'ucb': 0.8865928243658935},
  best: {'index': 4, 'name': '4', 'n_root': 3195776, 'win_flag': False, 'n': 1754880, 'n_wins': 1721240, 'q': 0.980830598103574, 'ucb': 0.9866734332771923}
}]
[performance info:
{
  steps: 2268,
  steps_per_second: 453.20508967358137,
  playouts: 3195776,
  playouts_per_second: 638598.7427939503,
  times_[ms]: {'total': 5004.356861114502, 'loop': 5000.383615493774, 'reduce_over_trees': 0.9992122650146484, 'reduce_over_actions': 0.9791851043701172, 'mean_loop': 2.2047546805528104, 'mean_select': 0.23814662633959788, 'mean_expand': 0.45445472054590835, 'mean_playout': 1.0848981993538993, 'mean_backup': 0.4192572842619827},
  trees: {'count': 8, 'mean_depth': 4.948057713651498, 'max_depth': 10, 'mean_size': 1126.25, 'max_size': 2322}
}]
MCTSNC RUN DONE. [time: 5.004356861114502 s; best action: 4, best win_flag: False best n: 1754880, best n_wins: 1721240, best q: 0.980830598103574]
BEST ACTION: 4
```

## Documentation
Complete developer documentation of the project is accessible at: [https://pklesk.github.io/mcts_numba_cuda](https://pklesk.github.io/mcts_numba_cuda). <br/>
Documentation for the `MCTSNC` class alone is at: [https://pklesk.github.io/mcts_numba_cuda/mctsnc.html](https://pklesk.github.io/fast_rboost_bins/mctsnc.html).
