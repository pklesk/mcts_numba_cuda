Search.setIndex({docnames:["c4","game_runner","gomoku","index","main","mcts","mctsnc","mctsnc_game_specifics","modules","utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["c4.rst","game_runner.rst","gomoku.rst","index.rst","main.rst","mcts.rst","mctsnc.rst","mctsnc_game_specifics.rst","modules.rst","utils.rst"],objects:{"":{c4:[0,0,0,"-"],game_runner:[1,0,0,"-"],gomoku:[2,0,0,"-"],main:[4,0,0,"-"],mcts:[5,0,0,"-"],mctsnc:[6,0,0,"-"],mctsnc_game_specifics:[7,0,0,"-"],utils:[9,0,0,"-"]},"c4.C4":{M:[0,2,1,""],N:[0,2,1,""],SYMBOLS:[0,2,1,""],__init__:[0,3,1,""],__module__:[0,2,1,""],__str__:[0,3,1,""],action_index_to_name:[0,3,1,""],action_name_to_index:[0,3,1,""],class_repr:[0,3,1,""],compute_outcome_job:[0,3,1,""],compute_outcome_job_numba_jit:[0,3,1,""],expand:[0,3,1,""],expand_one_random_child:[0,3,1,""],get_board:[0,3,1,""],get_board_shape:[0,3,1,""],get_extra_info:[0,3,1,""],get_extra_info_memory:[0,3,1,""],get_max_actions:[0,3,1,""],take_action_job:[0,3,1,""]},"game_runner.GameRunner":{OUTCOME_MESSAGES:[1,2,1,""],__init__:[1,3,1,""],__module__:[1,2,1,""],run:[1,3,1,""]},"gomoku.Gomoku":{M:[2,2,1,""],N:[2,2,1,""],SYMBOLS:[2,2,1,""],__init__:[2,3,1,""],__module__:[2,2,1,""],__str__:[2,3,1,""],action_index_to_name:[2,3,1,""],action_name_to_index:[2,3,1,""],class_repr:[2,3,1,""],compute_outcome_job:[2,3,1,""],compute_outcome_job_numba_jit:[2,3,1,""],expand:[2,3,1,""],expand_one_random_child:[2,3,1,""],get_board:[2,3,1,""],get_board_shape:[2,3,1,""],get_extra_info:[2,3,1,""],get_extra_info_memory:[2,3,1,""],get_max_actions:[2,3,1,""],take_action_job:[2,3,1,""]},"mcts.MCTS":{DEFAULT_SEARCH_STEPS_LIMIT:[5,2,1,""],DEFAULT_SEARCH_TIME_LIMIT:[5,2,1,""],DEFAULT_SEED:[5,2,1,""],DEFAULT_UCB_C:[5,2,1,""],DEFAULT_VANILLA:[5,2,1,""],DEFAULT_VERBOSE_DEBUG:[5,2,1,""],DEFAULT_VERBOSE_INFO:[5,2,1,""],__init__:[5,3,1,""],__module__:[5,2,1,""],__repr__:[5,3,1,""],__str__:[5,3,1,""],_backup:[5,3,1,""],_best_action:[5,3,1,""],_best_action_ucb:[5,3,1,""],_expand:[5,3,1,""],_make_actions_info:[5,3,1,""],_make_performance_info:[5,3,1,""],_playout:[5,3,1,""],_reduce_over_actions:[5,3,1,""],_select:[5,3,1,""],run:[5,3,1,""]},"mcts.State":{__init__:[5,3,1,""],__module__:[5,2,1,""],_subtree_depths:[5,3,1,""],_subtree_max_depth:[5,3,1,""],_subtree_size:[5,3,1,""],action_index_to_name:[5,3,1,""],action_name_to_index:[5,3,1,""],class_repr:[5,3,1,""],compute_outcome:[5,3,1,""],compute_outcome_job:[5,3,1,""],expand:[5,3,1,""],expand_one_random_child:[5,3,1,""],get_board:[5,3,1,""],get_board_shape:[5,3,1,""],get_extra_info:[5,3,1,""],get_extra_info_memory:[5,3,1,""],get_max_actions:[5,3,1,""],take_action:[5,3,1,""],take_action_job:[5,3,1,""]},"mctsnc.MCTSNC":{DEFAULT_DEVICE_MEMORY:[6,2,1,""],DEFAULT_N_PLAYOUTS:[6,2,1,""],DEFAULT_N_TREES:[6,2,1,""],DEFAULT_SEARCH_STEPS_LIMIT:[6,2,1,""],DEFAULT_SEARCH_TIME_LIMIT:[6,2,1,""],DEFAULT_SEED:[6,2,1,""],DEFAULT_UCB_C:[6,2,1,""],DEFAULT_VARIANT:[6,2,1,""],DEFAULT_VERBOSE_DEBUG:[6,2,1,""],DEFAULT_VERBOSE_INFO:[6,2,1,""],MAX_N_PLAYOUTS:[6,2,1,""],MAX_N_TREES:[6,2,1,""],MAX_STATE_BOARD_SHAPE:[6,2,1,""],MAX_STATE_EXTRA_INFO_MEMORY:[6,2,1,""],MAX_STATE_MAX_ACTIONS:[6,2,1,""],MAX_TREE_DEPTH:[6,2,1,""],MAX_TREE_SIZE:[6,2,1,""],VARIANTS:[6,2,1,""],__init__:[6,3,1,""],__module__:[6,2,1,""],__repr__:[6,3,1,""],__str__:[6,3,1,""],_backup_1_acp_prodigal:[6,3,1,""],_backup_1_acp_thrifty:[6,3,1,""],_backup_2_acp:[6,3,1,""],_backup_ocp:[6,3,1,""],_expand_1_acp_prodigal:[6,3,1,""],_expand_1_acp_thrifty:[6,3,1,""],_expand_1_ocp_prodigal:[6,3,1,""],_expand_1_ocp_thrifty:[6,3,1,""],_expand_2_prodigal:[6,3,1,""],_expand_2_thrifty:[6,3,1,""],_flatten_trees_actions_expanded_thrifty:[6,3,1,""],_json_dump:[6,3,1,""],_make_actions_info_prodigal:[6,3,1,""],_make_actions_info_thrifty:[6,3,1,""],_make_performance_info:[6,3,1,""],_memorize_root_actions_expanded:[6,3,1,""],_playout_acp_prodigal:[6,3,1,""],_playout_acp_thrifty:[6,3,1,""],_playout_ocp:[6,3,1,""],_reduce_over_actions_prodigal:[6,3,1,""],_reduce_over_actions_thrifty:[6,3,1,""],_reduce_over_trees_prodigal:[6,3,1,""],_reduce_over_trees_thrifty:[6,3,1,""],_reset:[6,3,1,""],_run_acp_prodigal:[6,3,1,""],_run_acp_thrifty:[6,3,1,""],_run_ocp_prodigal:[6,3,1,""],_run_ocp_thrifty:[6,3,1,""],_select:[6,3,1,""],_set_cuda_constants:[6,3,1,""],_validate_param:[6,3,1,""],init_device_side_arrays:[6,3,1,""],run:[6,3,1,""]},"utils.Logger":{__init__:[9,3,1,""],__module__:[9,2,1,""],flush:[9,3,1,""],write:[9,3,1,""]},c4:{C4:[0,1,1,""]},game_runner:{GameRunner:[1,1,1,""]},gomoku:{Gomoku:[2,1,1,""]},mcts:{MCTS:[5,1,1,""],State:[5,1,1,""]},mctsnc:{MCTSNC:[6,1,1,""]},mctsnc_game_specifics:{compute_outcome:[7,4,1,""],compute_outcome_c4:[7,4,1,""],compute_outcome_gomoku:[7,4,1,""],is_action_legal:[7,4,1,""],is_action_legal_c4:[7,4,1,""],is_action_legal_gomoku:[7,4,1,""],legal_actions_playout:[7,4,1,""],legal_actions_playout_c4:[7,4,1,""],legal_actions_playout_gomoku:[7,4,1,""],take_action:[7,4,1,""],take_action_c4:[7,4,1,""],take_action_gomoku:[7,4,1,""],take_action_playout:[7,4,1,""],take_action_playout_c4:[7,4,1,""],take_action_playout_gomoku:[7,4,1,""]},utils:{Logger:[9,1,1,""],cpu_and_system_props:[9,4,1,""],dict_to_str:[9,4,1,""],experiment_hash_str:[9,4,1,""],gpu_props:[9,4,1,""],hash_function:[9,4,1,""],hash_str:[9,4,1,""],list_to_str:[9,4,1,""],pickle_objects:[9,4,1,""],save_and_zip_experiment:[9,4,1,""],unpickle_objects:[9,4,1,""],unzip_and_load_experiment:[9,4,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"0":[5,6,9],"05":6,"1":6,"10":9,"128":6,"15":2,"16777216":6,"2":[5,6],"2048":6,"25":6,"256":6,"3":9,"31":9,"32":6,"4":[],"4096":6,"5":[5,6,9],"512":6,"6":0,"7":0,"8":6,"958041958041958":6,"byte":[],"case":[],"class":[0,1,2,3,5,6,9],"default":6,"float":[],"function":[3,6,9],"import":6,"int":[],"new":[],"return":[0,2,5,6,9],"static":[0,2,5,6],"true":[5,6],A:[],For:[3,6],In:6,It:[],The:3,With:6,__annotations__:[],__dict__:[],__doc__:[],__init__:[0,1,2,5,6,9],__main__:6,__module__:[0,1,2,5,6,9],__name__:6,__repr__:[5,6],__str__:[0,2,5,6],__weakref__:[],_backup:5,_backup_1_acp_prodig:6,_backup_1_acp_thrifti:6,_backup_2_acp:6,_backup_ocp:6,_best_act:5,_best_action_ucb:5,_decision_function_numba_cuda_job_int16:[],_expand:5,_expand_1_acp_prodig:6,_expand_1_acp_thrifti:6,_expand_1_ocp_prodig:6,_expand_1_ocp_thrifti:6,_expand_2_prodig:6,_expand_2_thrifti:6,_flatten_trees_actions_expanded_thrifti:6,_json_dump:6,_make_actions_info:5,_make_actions_info_prodig:6,_make_actions_info_thrifti:6,_make_performance_info:[5,6],_memorize_root_actions_expand:6,_playout:5,_playout_acp_prodig:6,_playout_acp_thrifti:6,_playout_ocp:6,_reduce_over_act:5,_reduce_over_actions_prodig:6,_reduce_over_actions_thrifti:6,_reduce_over_trees_prodig:6,_reduce_over_trees_thrifti:6,_reset:6,_run_acp_prodig:6,_run_acp_thrifti:6,_run_ocp_prodig:6,_run_ocp_thrifti:6,_select:[5,6],_set_cuda_const:6,_subtree_depth:5,_subtree_max_depth:5,_subtree_s:5,_validate_param:6,about:[],abov:6,absolut:[],acc:6,accord:[],acp_prodig:6,acp_thrifti:6,act:[],action:7,action_index:[0,2,5],action_index_to_nam:[0,2,5],action_index_to_name_funct:6,action_nam:[0,2,5],action_name_to_index:[0,2,5],action_ord:7,actions_info:5,actions_n:6,actions_ns_win:6,actions_win_flag:6,actual:6,addit:3,addition:6,after:[],algorithm:[],all:[],alloc:[],allow:[],along:[],also:3,among:6,an:9,ani:[],api:6,ar:[3,6],arg:[],argmax:[],arrai:6,associ:6,attribut:[],auxiliari:[3,9],avail:[],b:6,backup:[],base:[0,1,2,5,6,9],best:[],best_act:6,best_action_entri:5,best_n:6,best_n_win:6,best_win_flag:6,bin:[],binari:9,black:1,black_ai:1,block:[],board:[0,2,7],bool:[],boost:[],branch:[],bridg:[],budget:[],built:6,c4:[3,8],c_prop:9,call:[],callabl:[],can:6,captur:[],carlo:6,carri:[],castl:[],chess:[],child:[],children:5,choic:[],chosen:[],class_repr:[0,2,5],classes_:[],clf:6,clip:[],close:[],code:[6,9],com:[3,6,9],come:6,compil:6,comput:6,compute_outcom:[5,7],compute_outcome_c4:7,compute_outcome_gomoku:7,compute_outcome_job:[0,2,5],compute_outcome_job_numba_jit:[0,2],connect:[],consol:[],constant:[],constructor:[],contain:6,content:[],contract:[],convert:[],copi:6,core:[3,6],correct:[],correctli:6,correspond:[],cpu:9,cpu_and_system_prop:9,crucial:6,cuda:6,d:[5,9],data:[],dataset:6,debug:[],debug_verbos:[],decis:[],decision_function_mod:6,decision_function_numba_cuda_job_name_:[],decision_threshold_:[],decor:6,default_device_memori:6,default_n_playout:6,default_n_tre:6,default_se:[5,6],default_search_steps_limit:[5,6],default_search_time_limit:[5,6],default_ucb_c:[5,6],default_vanilla:5,default_vari:6,default_verbose_debug:[5,6],default_verbose_info:[5,6],defin:[],demonstr:3,depend:8,depth:5,describ:6,detail:[],dev_root_actions_expand:6,dev_trees_actions_expand:6,devic:[6,9],device_memori:6,dict_to_str:9,dictionari:9,digit:9,dimension:[],dispatch:6,docstr:6,document:6,doe:6,doubl:[],draw:1,driver:6,dtype:[],dtype_:[],dummi:[],dump:6,e2:[],e4:[],e:6,each:[],either:[],element:6,embodi:[3,6],en:[],end:[],env_hs_digit:9,equival:[],establish:[],estim:[],etc:[],exact:[],exampl:[3,8],execut:6,expand:[0,2,5],expand_one_random_child:[0,2,5],expans:9,expect:[],experi:9,experiment_h:9,experiment_hash_str:9,experiment_info:9,experiment_info_old:1,extra:[],extra_info:7,f:6,factor:[],fals:[5,6],fast_rboost_bin:6,faster:[],fastrealboostbin:6,featur:[],features_selected_:[],file:[3,6,9],first:[],fit:6,fit_mod:6,flag:[],float32:[],float64:[],flush:9,fname:[6,9],folder:9,follow:6,forced_search_steps_limit:[5,6],format:[6,9],formerli:[],fraction:[],frbb:6,friendli:[],from:[6,9],full:[],further:6,g:6,g_prop:9,game:[],game_class:1,game_index:1,game_runn:[3,8],gamerunn:1,gener:[],geq:6,get_board:[0,2,5],get_board_shap:[0,2,5],get_extra_info:[0,2,5],get_extra_info_memori:[0,2,5],get_max_act:[0,2,5],gigabyt:[],github:[3,6,9],given:9,gomoku:[3,8],gpu:[6,9],gpu_prop:9,hash:9,hash_funct:9,hash_str:9,high:6,host:6,http:[3,6,9],human:[],i:6,implement:6,impli:[],indent:9,independ:[],index:3,indic:[],inf:[5,6],inform:[3,9],inherit:6,init_device_side_arrai:6,input:[],insid:[],instal:8,instanc:[],int32:[],int64:[],int8:[],integ:9,intend:6,interv:[],investig:[],invoc:[],is_action_leg:7,is_action_legal_c4:7,is_action_legal_gomoku:7,iter:[],its:9,itself:[],jit:6,json:6,just:6,kernel:6,l:9,label:[],last:6,last_act:7,last_i:[0,2],last_j:[0,2],lead:6,learn:[],left:[],legal:[],legal_act:7,legal_actions_playout:7,legal_actions_playout_c4:7,legal_actions_playout_gomoku:7,legal_actions_with_count:7,leq:6,limit:[],link:8,list:9,list_to_str:9,load_breast_canc:6,logger:9,logit:[],logit_max:6,logits_:[],low:6,m:[0,2,7],machin:3,main:[3,8],main_hs_digit:9,mappingproxi:[],matchup_hs_digit:9,matchup_info:9,math:6,mathemat:6,max:[],max_n_playout:6,max_n_tre:6,max_state_board_shap:6,max_state_extra_info_memori:6,max_state_max_act:6,max_tree_depth:6,max_tree_s:6,maxes_selected_:[],maxim:[],maximum:[],mct:[0,2,6,8],mcts_numba_cuda:[3,9],mctsnc:[3,8],mctsnc_game_specif:[3,8],md:3,memor:[],memori:[],messag:9,method:[],minim:[],mins_selected_:[],mode:[],model_select:6,modul:[3,8],mont:6,more:3,most:[],multipl:[],must:6,n:[0,2,7],n_features_in_:[],n_game:1,n_playout:6,n_root_act:6,n_tree:6,name:6,ndarrai:[],necessari:[],node:[],none:[0,1,2,5,6],note:6,np:[],numba:6,numba_cuda:6,numba_jit:[],number:[],numer:[],numpi:6,nvidia:6,object:[1,5,6,9],ocp_prodig:6,ocp_thrifti:6,one:6,onli:[],open:[],oper:6,origin:[],os:9,other:[3,6],out:[],outcom:6,outcome_messag:1,outlier:[],outliers_ratio:6,output:6,over:[],page:3,parallel:6,param:9,paramet:[],parent:[0,2,5],passant:[],perform:6,pickl:9,pickle_object:9,pip:6,pklesk:[3,6,9],plai:3,player:[],playout:[],playout_root:5,pleas:3,pointer:[],possibl:[],power:[],predict:[],present:[3,6],print:6,privat:6,process:6,prodig:[],produc:6,progress:[],project:[3,8],properti:9,provid:[],ptype:6,purpos:6,python:6,random_generators_expand_1:6,random_generators_playout:6,random_st:6,rang:[],readm:3,reduct:[],refer:[],regist:[],relev:[],repositori:[3,8],repr:[5,6],repres:6,represent:9,requir:6,reset:[],respons:[],result:[],return_x_i:6,right:[],role:3,root:5,root_actions_expand:6,root_actions_info:5,root_board:6,root_children:5,root_extra_info:6,root_n:6,root_turn:6,round:[],run:[1,5,6],s:9,save:6,save_and_zip_experi:9,scikit:[],score:6,script:6,search:6,search_steps_limit:[5,6],search_time_limit:[5,6],second:[],see:[3,6],seed:[5,6],select:[],self:[0,2,5,6],set:9,shape:[],side:6,simpl:9,singl:6,skip:[],sklearn:6,so:[],some:6,some_list:9,sourc:[0,1,2,5,6,7,9],specif:[],sphinx:6,src:3,stage:[],standard:[],start:[],state:[0,2,5],state_board_shap:6,state_extra_info_memori:6,state_max_act:6,staticmethod:[],step:[],str:[0,2,5,6],stratifi:6,string:9,subsequ:[],substag:[],suitabl:[],sum:[],symbol:[0,2],system:6,t:6,take_act:[5,7],take_action_c4:7,take_action_gomoku:7,take_action_job:[0,2,5],take_action_playout:7,take_action_playout_c4:7,take_action_playout_gomoku:7,technic:[],test:6,test_siz:6,text:6,them:6,thi:[3,6],thorough:6,those:6,threshold:[],thrifti:[],time:6,todo:[],tool:6,train:6,train_test_split:6,transform:[],tree:6,trees_actions_expand:6,trees_actions_expanded_flat:6,trees_board:6,trees_depth:6,trees_extra_info:6,trees_leav:6,trees_n:6,trees_nodes_select:6,trees_ns_win:6,trees_outcom:6,trees_playout_outcom:6,trees_playout_outcomes_children:6,trees_selected_path:6,trees_siz:6,trees_termin:6,trees_turn:6,tupl:[],turn:[0,2,7],two:[],type:[],ucb_c:[5,6],uint64:[],uint8:[],underscor:6,unpickle_object:9,unzip_and_load_experi:9,us:[],usag:8,user:[],util:[3,8],valid:[],valu:[],vanilla:5,variabl:[],variant:6,verbos:[],verbose_debug:[5,6],verbose_info:[5,6],vertic:9,via:6,wa:6,weak:[],when:[],which:6,white:1,white_ai:1,win:1,within:[],write:[6,9],x:6,x_test:6,x_train:6,y:6,y_test:6,y_train:6},titles:["c4 module","game_runner module","gomoku module","Welcome to MCTS-NC\u2019s documentation!","main module","mcts module","mctsnc module","mctsnc_game_specifics module","src","utils module"],titleterms:{A:3,c4:0,carlo:3,content:3,cuda:3,depend:6,document:3,exampl:6,game_runn:1,gomoku:2,gpu:3,implement:3,indic:3,instal:6,link:[6,9],main:4,mct:[3,5],mctsnc:6,mctsnc_game_specif:7,modul:[0,1,2,4,5,6,7,9],mont:3,nc:3,numba:3,parallel:3,project:[6,9],python:3,repositori:[6,9],s:3,search:3,src:8,tabl:3,thorough:3,tree:3,usag:6,util:9,via:3,welcom:3}})