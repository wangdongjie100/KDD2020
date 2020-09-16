# Incremental Mobile User Profiling: Reinforcement Learning with Spatial Knowledge Graph for Modeling Event Streams

This repository provides a PyTorch implementation of the dynamic user profiling technique presented in our KDD 2020 paper.

## Abstract 

>We study the integration of reinforcement learning and spatial knowledge graph for incremental mobile user profiling,  which aims to map mobile users to dynamically-updated profile vectors by incremental learning from a mixed-user event stream. 
After exploring many profiling methods, we identify a new imitation based criteria to better evaluate and optimize profiling accuracy.
Considering the objective of teaching an autonomous agent to imitate a mobile user to plan next-visit based on the user's profile, the user profile is the most accurate when the agent can perfectly mimic the activity patterns of the user.
We propose to formulate the problem into a  reinforcement learning task, where an agent is a next-visit planner, an action is a POI that a user will visit next, and the state of environment is a fused representation of a user and spatial entities (e.g., POIs, activity types, functional zones).
An event that a user takes an action to visit a POI, will change the environment, resulting into a new state of user profiles and spatial entities, which helps the agent to predict next visit more accurately.
After analyzing such interactions among events, users, and spatial entities, we identify (1) semantic connectivity among spatial entities, and, thus, introduce a spatial Knowledge Graph (KG) to characterize the semantics of user visits over connected locations, activities, and zones. 
Besides, we identify (2) mutual influence between users and the spatial KG, and, thus, develop a mutual-updating strategy between users and the spatial KG, mixed with temporal context, to quantify the state representation that evolves over time.
Along these lines, we develop a reinforcement learning framework integrated with spatial KG. 
The proposed framework can achieve incremental learning in multi-user profiling given a mixed-user event stream.
Finally, we apply our approach to human mobility activity prediction and present extensive experiments to demonstrate improved performances.


## User Guide

We recommend you to install anaconda as your basic python library, here our code is running on python 3.7.4, then you can use this code according to the following steps.

1. Download our repository to your local machine and directory of choice:

`git clone  https://github.com/wangdongjie100/KDD2020.git`

2. Unzip the dataset. If you use windows or mac, you can choose double click "toy_data.zip", or you can type the following command in your terminal.

`unzip toy_data.zip`

3. We recommend you to create one specific environment for this code, if you do not care about this issue, you can skip to step 5.

`conda create -n kddtest python=3.7.4`

4. Activate the python environment.

`conda activate kddtest`

5. Install all required python packages, we have provided the requirements file, so you can use following command.

`pip install -r requirements.txt`

6. Run this code.

`python main.py`

If you want to change the experiment parameters, you can use following commands:

Beijing Experiment:

`python main.py --priority_mode='r' --city='bj' --model_name='dqn' --reward_mode='r1' --ll=0.3 --lc=0.3 --lp=0.4 --memory_capacity=20 --batch_size=6 --lr=0.001 --epsilon=0.9 --gamma=0.9 --target_replace_iter=5 --data_batch_size=1024 --user_path='toy_data/bj/s_user.pkl' --poi_dist_mat_path='toy_data/bj/poi_dist_Mat.pkl' --cat_sim_mat_path='toy_data/bj/cat_sim_mat.pkl' --s_KG_path='toy_data/bj/s_KG.pkl' --poi_cat_dict_path='toy_data/bj/POI_cat_dict.pkl' --poi_loc_dict_path='toy_data/bj/POI_loc_dict.pkl' --poi_list_train_path='toy_data/bj/POI_list_train.pkl' --user_list_train_path='toy_data/bj/user_list_train.pkl' --temporal_train_path='toy_data/bj/Temporal_train.pkl' --poi_list_test_path='toy_data/bj/POI_list_test.pkl' --user_list_test_path='toy_data/bj/user_list_test.pkl' --temporal_test_path='toy_data/bj/Temporal_test.pkl'`

NewYork Experiment:

`python main.py --priority_mode='r' --city='nyc' --model_name='dqn' --reward_mode='r1' --ll=0.3 --lc=0.3 --lp=0.4 --memory_capacity=20 --batch_size=6 --lr=0.001 --epsilon=0.9 --gamma=0.9 --target_replace_iter=5 --data_batch_size=1024 --user_path='toy_data/nyc/s_user.pkl' --poi_dist_mat_path='toy_data/nyc/poi_dist_Mat.pkl' --cat_sim_mat_path='toy_data/nyc/cat_sim_mat.pkl' --s_KG_path='toy_data/nyc/s_KG.pkl' --poi_cat_dict_path='toy_data/nyc/POI_cat_dict.pkl' --poi_loc_dict_path='toy_data/nyc/POI_loc_dict.pkl' --poi_list_train_path='toy_data/nyc/POI_list_train.pkl' --user_list_train_path='toy_data/nyc/user_list_train.pkl' --temporal_train_path='toy_data/nyc/Temporal_train.pkl' --poi_list_test_path='toy_data/nyc/POI_list_test.pkl' --user_list_test_path='toy_data/nyc/user_list_test.pkl' --temporal_test_path='toy_data/nyc/Temporal_test.pkl'`

## Reference 
If you use this code or our work for your research, please cite our paper.

```
@inproceedings{wang2020incremental,
  title={Incremental Mobile User Profiling: Reinforcement Learning with Spatial Knowledge Graph for Modeling Event Streams},
  author={Wang, Pengyang and Liu, Kunpeng and Jiang, Lu and Li, Xiaolin and Fu, Yanjie},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={853--861},
  year={2020}
}
```

