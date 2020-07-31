import functools
import time
import torch
import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)



def log_func_time(func):
    @functools.wraps(func)
    def wrapper(*args,**kw):
        logging.info("Begin running : %s" % func.__name__)
        old_time = time.time()
        result = func(*args,**kw)
        logging.info("%s spent for %.4f s!" % (func.__name__,time.time()-old_time))
        logging.info("End running: %s \n",func.__name__)
        return result
    return wrapper


@log_func_time
def load_public_data(params):
    # load poi cat dict
    with open(params.poi_cat_dict_path, "rb") as f:
        poi_cat_dict = pickle.load(f)
    logging.info("Loading poi category dict done!")

    # load poi loc dict
    with open(params.poi_loc_dict_path, "rb") as f:
        poi_loc_dict = pickle.load(f)
    logging.info("Loading poi location dict done!")

    # load poi dist mat
    with open(params.poi_dist_mat_path, "rb") as f:
        poi_dist_mat = pickle.load(f)
    logging.info("Loading poi distance mat done!")

    # load cat sim mat
    with open(params.cat_sim_mat_path, "rb") as f:
        cat_sim_mat = pickle.load(f)
    logging.info("Loading poi category similarity done!")

    #load user embedding initialization
    with open(params.user_path,"rb") as f:
        s_u = pickle.load(f)
    logging.info("Loading initial user profiling done!")

    #load spatial KG embedding initialization
    with open(params.s_KG_path,"rb") as f:
        s_KG = pickle.load(f)
    logging.info("Loading intial spatial KG embedding done!")

    return poi_dist_mat, cat_sim_mat, poi_cat_dict, poi_loc_dict, s_u, s_KG

@log_func_time
def load_train_data(params):
    #load POI sequence for training
    with open(params.poi_list_train_path,"rb") as f:
        poi_list_train = pickle.load(f)
    logging.info("Loading poi sequence training done!")

    #load user list for training
    with open(params.user_list_train_path,"rb") as f:
        user_list_train = pickle.load(f)
    logging.info("Loading user list training done!")

    #load temporal context for training
    with open(params.temporal_train_path,"rb") as f:
        temporal_context_train = pickle.load(f)
    logging.info("Loading temporal context training done!")

    user_list_train = torch.tensor(np.array(user_list_train))
    poi_list_train = torch.tensor(np.array(poi_list_train))
    temporal_context_train = torch.tensor(np.array(temporal_context_train))


    return poi_list_train, user_list_train, temporal_context_train

@log_func_time
def load_test_data(params):
    # load POI sequence for testing
    with open(params.poi_list_test_path,"rb") as f:
        poi_list_test = pickle.load(f)
    logging.info("Loading poi sequence training done!")

    # load user list for testing
    with open(params.user_list_test_path,"rb") as f:
        user_list_test = pickle.load(f)
    logging.info("Loading user list training done!")

    # load temporal context for testing
    with open(params.temporal_test_path,"rb") as f:
        temporal_context_test = pickle.load(f)
    logging.info("Loading temporal context training done!")

    user_list_test = torch.tensor(np.array(user_list_test))
    poi_list_test = torch.tensor(np.array(poi_list_test))
    temporal_context_test = torch.tensor(np.array(temporal_context_test))

    return poi_list_test,user_list_test, temporal_context_test



def reward(lambda_l, lambda_c, lambda_p, predict_POI_index, real_POI_index, POI_cat_dict, POI_loc_dict, poi_dist_mat, cat_sim_mat):
    '''
    reward function
        r = \lambda_l * distance_between_real_estimated_POI + \
            \lambda_c * similarity_between_real_estimated_POI_category + \
            \lambda_p * (whether_estimated_POI_is_the_real_one)

    :param lambda_l: float, weight for distance_between_real_estimated_POI
    :param lambda_c: float, weight for similarity_between_real_estimated_POI_category
    :param lambda_p: float, weight for whether_estimated_POI_is_the_real_one
    :param action_index: int, estimated POI
    :param POI_loc_dict: dict, POI-location mapping dict
    :param POI_cat_dict: dict, POI-category mapping dict
    :param POI_list: list, real POI visiting list
    :param poi_dist_mat: scipy.sparse.csr.csr_matrix, distance maxtrix between POIs
    :param cat_sim_mat: scipy.sparse.csr.csr_matrix, similarity maxtrix between POI categories
    :param visit_counter: int, counter for remembering the poisition of visit event.
    :return: reward: float, reward value for the given action
    '''


    POI_left = min(POI_loc_dict[predict_POI_index], POI_loc_dict[real_POI_index])
    POI_right = max(POI_loc_dict[predict_POI_index], POI_loc_dict[real_POI_index])

    real_cat_index = POI_cat_dict[real_POI_index]
    cat_index = POI_cat_dict[predict_POI_index]
    cat_left = min(cat_index, real_cat_index)
    cat_right = max(cat_index, real_cat_index)

    distance = poi_dist_mat[POI_left, POI_right]
    if distance == 0:
        distance = 0
    else:
        distance = 1 / distance

    r = lambda_l * distance + \
        lambda_c * cat_sim_mat[cat_left, cat_right] + \
        lambda_p * [1 if predict_POI_index == real_POI_index else 0][0]
    return r


def evaluate_model(y, y_pred, POI_cat_dict, POI_loc_dict, dist_mat, sim_mat):

    y_cat_true = list(map(lambda x: POI_cat_dict[x], y))
    y_cat_pred = list(map(lambda x: POI_cat_dict[x], y_pred))

    cat_df = pd.DataFrame({'true':y_cat_true, 'pred':y_cat_pred})
    cat_df['left'] = cat_df.apply(lambda x: min(x['true'], x['pred']), axis=1)
    cat_df['right'] = cat_df.apply(lambda x: max(x['true'], x['pred']), axis=1)
    cat_sim_all = cat_df.apply(lambda x: sim_mat[x['left'], x['right']], axis=1)

    y_dist_true = list(map(lambda x: POI_loc_dict[x], y))
    y_dist_pred = list(map(lambda x: POI_loc_dict[x], y_pred))
    dist_df = pd.DataFrame({'true':y_dist_true, 'pred':y_dist_pred})
    dist_df['left'] = dist_df.apply(lambda x: min(x['true'], x['pred']), axis=1)
    dist_df['right'] = dist_df.apply(lambda x: max(x['true'], x['pred']), axis=1)
    distance = dist_df.apply(lambda x: dist_mat[x['left'], x['right']], axis=1)



    cat_precision = precision_score(y_cat_true, y_cat_pred, average='weighted')
    cat_recall = recall_score(y_cat_true, y_cat_pred, average='weighted')
    avg_dist = sum(distance) / len(distance)
    avg_sim = sum(cat_sim_all) / len(cat_sim_all)

    return cat_precision, cat_recall, avg_sim, avg_dist




