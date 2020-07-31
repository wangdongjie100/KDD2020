import torch
import numpy as np


class Environment(torch.nn.Module):
    """
    Implementation for environment to update states
    """
    def __init__(self, n_state, n_t_1, n_t_2):
        '''
        Initialization
        :param n_h: int, dimension of weights/state
        '''
        super(Environment, self).__init__()
        self.n_state = n_state
        self.n_t_1 = n_t_1
        self.n_t_2 = n_t_2

        self.W_u = torch.nn.Parameter(data=torch.zeros((self.n_state, 1)), requires_grad=True)
        self.W_p = torch.nn.Parameter(data=torch.zeros((self.n_state, 1)), requires_grad=True)
        self.W_T_1 = torch.nn.Parameter(data=torch.zeros((self.n_state, self.n_t_1)), requires_grad=True)
        self.W_T_2 = torch.nn.Parameter(data=torch.zeros((self.n_t_2, 1)), requires_grad=True)
        self.b_T = torch.nn.Parameter(data=torch.zeros((self.n_state, 1)), requires_grad=True)
        self.W_p_ = torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)


        self.W_u.data.uniform_(-1, 1)
        self.W_p.data.uniform_(-1, 1)
        self.W_T_1.data.uniform_(-1, 1)
        self.W_T_2.data.uniform_(-1, 1)
        self.b_T.data.uniform_(-1,1)
        self.W_p_.data.uniform_(-1,1)






    def forward(self, s_u, s_KG, T, user_index, POI_index):
        '''
        Update user state:
            \mathbf{u}_i^{l+1} = \sigma(\mathbf{u}_i^{l} + \mathbf{W}_u \cdot (\mathbf{h}_{P_j}^{l})^{\intercal} \cdot  \mathbf{T}^l)

        Update KG state:
            1> update visited POI:
                \mathbf{h}_{P_j}^{l+1} = \sigma(\mathbf{h}_{P_j}^{l} + \mathbf{W}_p \cdot (\mathbf{u}_{i}^{l})^{\intercal} \cdot  \mathbf{T}^l)
            2> update neighboring entities:
                \mathbf{t}_{\ast}^{l+1} = \mathbf{h}_{P_j}^{l+1} + \mathbf{r}_{(P_j,\ast)}
        :param s_u: torch.tensor, num_user*dimension, user state tensor
        :param s_KG: torch_geometric.data.Data, KG state
        :param T: torch.tensor, num_step*dimension_traffic, temporal context tensor represented by traffic in each geo-grid
        :param user_index: int, which user to visit
        :param POI_index: int, which POI is visited
        :return: s_u_, s_KG_, updated user and spatial KG states
        '''


        s_u_ = s_u
        s_KG_ = s_KG

        T_transformed = torch.sigmoid(torch.mm(torch.mm(self.W_T_1, T), self.W_T_2)+self.b_T)

        cur_user = s_u[user_index].clone()
        cur_POI = s_KG.x[POI_index].clone()

        s_u_[user_index] = torch.sigmoid(cur_user + \
                                     torch.mm(torch.mm(self.W_u, cur_POI.view(1, -1)), T_transformed).view(-1))

        s_KG_.x[POI_index] = torch.sigmoid(cur_POI + \
                                     torch.mm(torch.mm(self.W_p, cur_user.view(1, -1)), T_transformed).view(-1))

        rel_index = s_KG.edge_index[0]==POI_index
        tail_index = s_KG.edge_index[1][rel_index]
        s_KG_.x[tail_index] = s_KG_.x[POI_index] + s_KG_.edge_attr[rel_index]


        #update the embedding of the neighborhoods that have same category and the same location POIs
        cate_func = tail_index.numpy().tolist()
        neigh_index = []
        neigh_rel = np.zeros(s_KG.edge_index.shape[1],dtype=bool)
        for item in cate_func:
            neigh_rel_index = s_KG.edge_index[1] == item
            neigh_poi_index = s_KG.edge_index[0][neigh_rel_index].numpy().tolist()
            neigh_index.extend(neigh_poi_index)
            neigh_rel = neigh_rel | neigh_rel_index.numpy()
        neigh_index.remove(POI_index)
        neigh_rel[POI_index] = False

        neigh_old_poi_emb = s_KG.x[neigh_index]
        neigh_new_poi_emb = s_KG_.x[neigh_index] - s_KG_.edge_attr[neigh_rel]

        neigh_value = neigh_old_poi_emb + torch.mm(self.W_p_,neigh_new_poi_emb.T).T
        s_KG_.x[neigh_index] = torch.sigmoid(neigh_value.view(-1)).view(neigh_value.shape[0],neigh_value.shape[1])

        return s_u_, s_KG_