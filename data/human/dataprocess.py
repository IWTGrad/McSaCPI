import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from collections import defaultdict
from utils import *
from PyBioMed.PyProtein import CTD

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index


def create_data():
    data_path = '.'
    drugdata_file = data_path + '/human_drug.txt'
    prodataw_file = data_path + '/human_protein.txt'
    data_file = data_path + '/human_data.csv'

    smile_graph = {}
    with open(drugdata_file) as f:
       smiles = f.read().strip().split('\n')
       for smile in smiles:
           smile_graph[smile]=smile_to_graph(smile)
    protein_CTD = {}
    with open(prodataw_file) as f:
        proteins = f.read().strip().split('\n')
        for protein in proteins:
            protein_CTD[protein] = list(CTD.CalculateCTD(protein).values())
    pairs = pd.read_csv(data_file, index_col=None)
    #train_nums = np.load('./data/train_nums.npy')
    #test_nums = np.load('./data/test_nums.npy')
    #np.random.shuffle(train_nums)
    #np.random.shuffle(test_nums)
    pairs=np.asarray(pairs)
    np.random.shuffle(pairs)

    train_data = pairs[0:int(len(pairs)*0.8)]
    test_data = pairs[int(len(pairs) * 0.8):len(pairs)]

    pro_embedding = torch.load('./protein_embedding/human_ProtTransAlbertBFDEmbedder_Pro_Embedding.pt')
    drug_embedding = torch.load('./drug_embedding/human_MolecularTransformerEmbeddings.pt')
    train_drug, train_protein, train_interaction = np.asarray(train_data.T[0]), \
                                                   np.asarray(train_data.T[1]),np.asarray(train_data.T[2])
    train_datasets = DTADataset_CTD(root='data/CTD', dataset='human_ProtTransAlbertBFDEmbedder_MolecularTransformer_train', drug=train_drug,
                                interaction=train_interaction, smile_graph=smile_graph,
                                target_protein=train_protein, pro_embedding=pro_embedding,pro_CTD=protein_CTD,
                                drug_embedding=drug_embedding)

    test_drug, test_protein, test_interaction = np.asarray(test_data.T[0]),\
                                                np.asarray(test_data.T[1]), np.asarray(test_data.T[2])
    test_datasets = DTADataset_CTD(root='data/CTD', dataset='human_ProtTransAlbertBFDEmbedder_MolecularTransformer_test', drug=test_drug,
                               interaction=test_interaction, smile_graph=smile_graph,
                               target_protein=test_protein, pro_embedding=pro_embedding,pro_CTD=protein_CTD,
                               drug_embedding=drug_embedding)


if __name__ == "__main__":
    create_data()
