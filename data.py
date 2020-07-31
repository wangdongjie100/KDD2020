import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from utils import load_public_data,load_train_data,load_test_data

import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class User_Activity_Batch(object):
    def __init__(self,data):
        transposed_data = list(zip(*data))
        self.user_list = torch.stack(transposed_data[0],0)
        self.poi_list = torch.stack(transposed_data[1],0)
        self.temporal_list = torch.stack(transposed_data[2],0)

def collate_wrapper(batch):
    return User_Activity_Batch(batch)


class POI_Info(object):
    poi_dist_mat = ""
    cat_sim_mat = ""
    poi_cat_dict = ""
    poi_loc_dict = ""
    env_state = ""

    def __init__(self,*data):
        POI_Info.poi_dist_mat = data[0]
        POI_Info.cat_sim_mat = data[1]
        POI_Info.poi_cat_dict = data[2]
        POI_Info.poi_loc_dict = data[3]
        POI_Info.env_nt_1 = data[4]
        POI_Info.env_nt_2 = 3

class User_KG(object):
    s_u = ""
    s_KG = ""

    def __init__(self,*data):
        User_KG.s_u = data[0]
        User_KG.s_KG = data[1]




def load_data(params):

    '''
    load data from pkl files. The data can be divided into three types:
    1. public data: this is the public attribute data for the experiment
    2. train  data: this is used to train model
    3. test data: this is used to test the model function
    :param params:
    :return:
    '''

    poi_dist_mat, cat_sim_mat, poi_cat_dict, poi_loc_dict, s_u, s_KG = load_public_data(params)

    poi_list_train, user_list_train, temporal_context_train = load_train_data(params)

    poi_list_test,user_list_test, temporal_context_test = load_test_data(params)

    # unify the data digit type
    temporal_context_train = temporal_context_train.float()
    temporal_context_test = temporal_context_test.float()
    s_u = s_u.float()
    s_KG.x = s_KG.x.float()
    s_KG.edge_attr = s_KG.edge_attr.float()

    train_dataset = TensorDataset(user_list_train,poi_list_train,temporal_context_train)
    test_dataset = TensorDataset(user_list_test,poi_list_test,temporal_context_test)

    train_loader = DataLoader(train_dataset,batch_size=params.data_batch_size,collate_fn=collate_wrapper,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=params.data_batch_size,collate_fn=collate_wrapper,pin_memory=True)

    poi_info =  POI_Info(poi_dist_mat, cat_sim_mat, poi_cat_dict, poi_loc_dict,temporal_context_train[0].view(-1, 3).shape[0])
    user_KG = User_KG(s_u,s_KG)

    return train_loader, test_loader, poi_info, user_KG









