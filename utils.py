import os
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
from torch import optim
from math import sqrt
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, precision_score, recall_score,roc_curve


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='DUDE',
                 drug=None, interaction=None, transform=None,
                 pre_transform=None, smile_graph=None, target_protein=None, pro_embedding=None, drug_embedding=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data_mol = torch.load(self.processed_paths[0])

        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(drug, target_protein, interaction, smile_graph, pro_embedding, drug_embedding)
            self.data_mol = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug, target_protein, y, smile_graph, pro_embedding, drug_embedding):
        assert (len(drug) == len(y)), 'The three lists must be the same length!'
        data_list = []

        data_len = len(drug)
        for i in range(data_len):
            smiles = drug[i]

            labels = y[i]
            protein = target_protein[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]

            # print(np.array(features).shape, np.array(edge_index).shape)
            # print(target_features.shape, target_edge_index.shape)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.LongTensor([labels]))

            GCNData.smiles = drug_embedding[smiles].view(1, -1)

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData.proseq = pro_embedding[protein].view(1, -1)

            # print(GCNData.target.size(), GCNData.target_edge_index.size(), GCNData.target_x.size())
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data = data_list

        torch.save(self.data, self.processed_paths[0])

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]

class DTADataset_CTD(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='DUDE',
                 drug=None, interaction=None, transform=None,
                 pre_transform=None ,smile_graph=None, target_protein=None, pro_CTD=None,pro_embedding=None, drug_embedding=None):

        super(DTADataset_CTD, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        #if os.path.isfile(self.processed_paths[0]):
        #    print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
        #    self.data_mol = torch.load(self.processed_paths[0])

        #else:
        print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
        self.process(drug, target_protein, interaction, smile_graph, pro_embedding,pro_CTD, drug_embedding)
        self.data_mol = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug, target_protein, y, smile_graph, pro_embedding, pro_CTD,drug_embedding):
        assert (len(drug) == len(y)), 'The three lists must be the same length!'
        data_list = []

        data_len = len(drug)
        for i in range(data_len):
            smiles = drug[i]

            labels = y[i]
            protein = target_protein[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]

            # print(np.array(features).shape, np.array(edge_index).shape)
            # print(target_features.shape, target_edge_index.shape)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.LongTensor([labels]))

            GCNData.smiles = drug_embedding[smiles].view(1, -1)

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData.proseq = pro_embedding[protein].view(1, -1)
            GCNData.pro_CTD=torch.FloatTensor(pro_CTD[protein]).view(1, -1)

            # print(GCNData.target.size(), GCNData.target_edge_index.size(), GCNData.target_x.size())
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data = data_list

        torch.save(self.data, self.processed_paths[0])

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]

class Trainer(object):
    def __init__(self, model, lr=0.0001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=0.001)

    def train(self, train_loader, device, epoch):
        print('Training on {} samples...'.format(len(train_loader.dataset)))
        self.model.train()
        loss_total = 0
        #if (epoch+1) % 20 == 0:
        #    self.optimizer.param_groups[0]['lr'] *= 0.5
        test = enumerate(train_loader)
        batch_idx=0
        for data in enumerate(train_loader):
            batch_idx+=1
            data = data.to(device)
            self.optimizer.zero_grad()
            loss = self.model(data)
            loss.backward()
            loss_total += loss
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss))
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, test_loader, device):
        self.model.eval()
        print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in test_loader:
                (correct_labels, predicted_labels,predicted_scores) = self.model(data.to(device), train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        RE1 = self.getROCE(Y, T, 0.5)
        RE2 = self.getROCE(Y, T, 1)
        RE3 = self.getROCE(Y, T, 2)
        RE4 = self.getROCE(Y, T, 5)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        fper, tper, thresholds = roc_curve(np.asarray(T), np.asarray(S))
        return AUC, precision, recall, RE1,RE2,RE3,RE4,fper, tper

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

    def getROCE(self,predList, targetList, roceRate):
        """
        getROCE(all_pred, all_target, 0.5)
        :param predList:
        :param targetList:
        :param roceRate:
        :return:
        """
        p = sum(targetList)  # 正样本数
        n = len(targetList) - p  # 负样本数
        predList = [[index, x] for index, x in enumerate(predList)]
        predList = sorted(predList, key=lambda x: x[1], reverse=True)
        tp1 = 0
        fp1 = 0
        maxIndexs = []
        for x in predList:
            if (targetList[x[0]] == 1):
                tp1 += 1
            else:
                fp1 += 1
                if (fp1 > ((roceRate * n) / 100)):
                    break
        roce = (tp1 * n) / (p * fp1)
        return roce
