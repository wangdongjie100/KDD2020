''' Options
This script is used to load the arg parameters for RLKGModel

Inputs : None

Returns:
    model parameters
'''

import argparse

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="RLKGModel parameters")

        #model parameter
        self.parser.add_argument("--priority_mode", type=str, default='r', help="priority mode")
        self.parser.add_argument("--city", type=str, default="bj", help="city name")
        self.parser.add_argument("--model_name", type=str, default="dqn", help="model name")
        self.parser.add_argument("--reward_mode", type=str, default="r1", help="reward mode")
        self.parser.add_argument("--ll", type=float, default=0.3, help="location ratio")
        self.parser.add_argument("--lc", type=float, default=0.3, help="category ratio")
        self.parser.add_argument("--lp", type=float, default=0.4, help="POI ratio")
        self.parser.add_argument("--memory_capacity", type=int, default=20, help="memory capacity")
        self.parser.add_argument("--batch_size", type=int, default=6, help="batch size")
        self.parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        self.parser.add_argument("--epsilon", type=float, default=0.9, help="epsilon")
        self.parser.add_argument("--gamma", type=float, default=0.9, help="gamma")
        self.parser.add_argument("--target_replace_iter", type=int, default=5, help="target replace iter")
        self.parser.add_argument("--data_batch_size", type=int, default=1024, help="This data batch size is for generating the data loader")

        #basic data parameter
        self.parser.add_argument("--user_path", type=str, default="toy_data/bj/s_user.pkl", help="user data path")
        self.parser.add_argument("--poi_dist_mat_path", type=str, default="toy_data/bj/poi_dist_Mat.pkl",
                            help="poi dis mat path")
        self.parser.add_argument("--cat_sim_mat_path", type=str, default="toy_data/bj/cat_sim_mat.pkl",
                            help="cat sim mat path")
        self.parser.add_argument("--s_KG_path", type=str, default="toy_data/bj/s_KG.pkl", help="s_KG path")
        self.parser.add_argument("--poi_cat_dict_path", type=str, default="toy_data/bj/POI_cat_dict.pkl",
                            help="poi cat dict path")
        self.parser.add_argument("--poi_loc_dict_path", type=str, default="toy_data/bj/POI_loc_dict.pkl",
                            help="poi loc dict path")

        #train data parameter
        self.parser.add_argument("--poi_list_train_path", type=str, default="toy_data/bj/POI_list_train.pkl",
                            help="poi list train path")
        self.parser.add_argument("--user_list_train_path", type=str, default="toy_data/bj/user_list_train.pkl",
                            help="user list train path")
        self.parser.add_argument("--temporal_train_path", type=str, default="toy_data/bj/Temporal_train.pkl",
                            help="temporal train path")

        #test data parameter
        self.parser.add_argument("--poi_list_test_path", type=str, default="toy_data/bj/POI_list_test.pkl",
                            help="poi list test path")
        self.parser.add_argument("--user_list_test_path", type=str, default="toy_data/bj/user_list_test.pkl",
                            help="user list test path")
        self.parser.add_argument("--temporal_test_path", type=str, default="toy_data/bj/Temporal_test.pkl",
                            help="temporal test path")

    def parse(self):
        return self.parser.parse_args()