'''
This script contains the main content of RLKGModel
'''

from dqn import DQN
from environment import Environment
from utils import reward,evaluate_model
import numpy as np
import logging
import torch
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

class RLKGModel(object):

    def __init__(self,poi_info,user_KG,params):
        self.poi_info = poi_info
        self.user_KG = user_KG
        self.visit_counter = 0
        self.ll = params.ll
        self.lc = params.lc
        self.lp = params.lp
        self.poi_cat_dict = poi_info.poi_cat_dict
        self.poi_loc_dict = poi_info.poi_loc_dict
        self.poi_dist_mat = poi_info.poi_dist_mat
        self.cat_sim_mat  = poi_info.cat_sim_mat

        self.memory_capacity = params.memory_capacity

        self.environment = Environment(user_KG.s_u.shape[1],
                                       self.poi_info.env_nt_1,
                                       self.poi_info.env_nt_2)

        self.dqn = DQN(self.environment,
                       user_KG.s_u.shape[1] + user_KG.s_KG.x.shape[1],
                       user_KG.s_KG.num_POI,
                       params.memory_capacity,
                       params.lr,
                       params.epsilon,
                       params.batch_size,
                       params.gamma,
                       params.target_replace_iter,
                       mode=params.priority_mode)

        self.predict_POI_index = np.random.randint(user_KG.s_KG.num_POI)

        self.r = reward(params.ll, params.lc, params.lp,
                        self.predict_POI_index, 0,
                        poi_info.poi_cat_dict, poi_info.poi_loc_dict,
                        poi_info.poi_dist_mat, poi_info.cat_sim_mat)





    def fit(self,train_loader):

        num_visits = len(train_loader) * train_loader.batch_size
        for batch_ndx, sample in enumerate(train_loader):
            for i in range(len(sample.user_list)):
                s_u_, s_KG_ = self.environment(self.user_KG.s_u, self.user_KG.s_KG,
                                               sample.temporal_list[i].view(self.poi_info.env_nt_1, self.poi_info.env_nt_2),
                                               sample.user_list[i],
                                               sample.poi_list[i])
                self.dqn.store_transition((self.user_KG.s_u[sample.user_list[i]], self.user_KG.s_KG),
                                           self.predict_POI_index, self.r,
                                          (s_u_[sample.user_list[i]], s_KG_))

                if self.dqn.memory_counter > self.memory_capacity:
                    self.dqn.learn()

                if self.visit_counter < num_visits - 1:
                    self.user_KG.s_u = s_u_
                    self.user_KG.s_KG = s_KG_
                    self.visit_counter += 1
                    self.predict_POI_index = self.dqn.choose_action((self.user_KG.s_u[sample.user_list[i]], self.user_KG.s_KG))
                    self.r = reward(self.ll, self.lc, self.lp,
                               self.predict_POI_index, int(sample.poi_list[self.visit_counter % (train_loader.batch_size*batch_ndx+1)]),
                               self.poi_cat_dict, self.poi_loc_dict,
                               self.poi_dist_mat, self.cat_sim_mat)

            logging.info("The {} training batch has been done!".format(batch_ndx))

    def evaluate(self,test_loader):
        preds = []
        for batch_ndx, sample in enumerate(test_loader):
            with torch.no_grad():
                for i in range(len(sample.user_list)):
                    action_index = self.dqn.choose_action((self.user_KG.s_u[sample.user_list[i]], self.user_KG.s_KG))
                    s_u_, s_KG_ = self.environment(self.user_KG.s_u, self.user_KG.s_KG,
                                                   sample.temporal_list[i].view(self.poi_info.env_nt_1, self.poi_info.env_nt_2),
                                                   sample.user_list[i],sample.poi_list[i])
                    self.user_KG.s_u = s_u_
                    self.user_KG.s_KG = s_KG_
                    preds.append(action_index)

        reals = [int(x) for x in test_loader.dataset.tensors[1]]
        cat_precision, cat_recall, avg_sim, avg_dist = evaluate_model(reals, preds,
                                                                      self.poi_cat_dict, self.poi_loc_dict,
                                                                      self.poi_dist_mat, self.cat_sim_mat)
        logging.info("Final cat_precision:{},cat_recall:{},avg_sim:{},avg_dist:{}".format(cat_precision, cat_recall, avg_sim,avg_dist))
