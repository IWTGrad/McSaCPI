import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv,GCNConv,GATConv,GatedGraphConv,ClusterGCNConv,GraphConv,GINConv,global_max_pool as gmp
import numpy as np


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_hkeads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class AttLayer(torch.nn.Module):
    def __init__(self,input_dim,attention_dim):
        self.W = torch.nn.Linear(input_dim, attention_dim, bias=True)
    def forward(self,pro_input,drug_input):
        h=torch.tanh(self.W(input))



# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=512, dropout=0.1):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.drug_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)

        #self.drug_gated_graph=GatedGraphConv(out_channels=num_features_xd * 4,num_layers=2)
        #self.drug_cluster1=ClusterGCNConv(in_channels=num_features_xd,out_channels=num_features_xd )
        #self.drug_cluster2 = ClusterGCNConv(in_channels=num_features_xd, out_channels=num_features_xd*2)
        #self.drug_cluster3 = ClusterGCNConv(in_channels=num_features_xd*2, out_channels=num_features_xd*4)


        self.drug_fc1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.drug_fc2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.pro_embedding_fc1 = nn.Linear(4096, 2048)
        self.pro_embedding_fc2 = nn.Linear(2048, output_dim)

        self.drug_embedding_fc1 = nn.Linear(512, 1024)
        self.drug_embedding_fc2 = nn.Linear(1024, output_dim)

        self.muti_attention=MultiHeadAttention(input_dim=output_dim*3,n_heads=4)
        # combined layers
        self.fc1 = nn.Linear(6*output_dim, 2*output_dim)
        self.fc2 = nn.Linear(2*output_dim, output_dim)

        self.out = nn.Linear(output_dim, 2)



    def forward(self, data):
        # get graph input
        batch=data.batch
        drug_edge_index, drug_graph_feature, drug_embedding, \
        pro_embedding, correct_interaction = data.edge_index, data.x, data.smiles, data.proseq, data.y

        x = self.drug_conv1(drug_graph_feature, drug_edge_index)
        x = self.relu(x)
        x = self.drug_conv2(x, drug_edge_index)
        x = self.relu(x)
        x = self.drug_conv3(x, drug_edge_index)
        x = self.relu(x)

        #x = self.drug_gated_graph(drug_graph_feature, drug_edge_index)
        #x = self.relu(x)

        #x = self.drug_cluster1(drug_graph_feature, drug_edge_index)
        #x = self.relu(x)
        #x = self.drug_cluster2(x, drug_edge_index)
        #x = self.relu(x)
        #x = self.drug_cluster3(x, drug_edge_index)
        #x = self.relu(x)

        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.drug_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.drug_fc2(x))
        x = self.dropout(x)

        # 1d conv layers
        pro_embedding=self.pro_embedding_fc1(pro_embedding)
        pro_embedding=self.relu(pro_embedding)
        pro_embedding = self.pro_embedding_fc2(pro_embedding)
        pro_embedding = self.relu(pro_embedding)
        # flatten
        drug_embedding=self.drug_embedding_fc1(drug_embedding)
        drug_embedding=self.relu(drug_embedding)
        drug_embedding = self.drug_embedding_fc2(drug_embedding)
        drug_embedding = self.relu(drug_embedding)
        # concat
        xc = torch.cat((x,pro_embedding,drug_embedding), 1)

        xc_attention=self.muti_attention(xc)
        xc=torch.cat((xc,xc_attention),1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

    def __call__(self, data, train=True):
        drug_edge_index,drug_graph_feature,drug_embedding,pro_embedding,correct_interaction=data.edge_index,data.x,data.smiles,\
                                                                                            data.proseq,data.y

        predicted_interaction = self.forward(data)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores
