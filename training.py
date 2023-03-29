import numpy as np
import torch
from torch_geometric.data import DataLoader
from model.GCN2 import GCNNet
from model.GAT import GATNet
import torch.nn.functional as F
from data.DUDE.utiles_test import *
import matplotlib.pyplot as plt

dataset='DUDE'
file_AUCs = './output/result/'+dataset+'_AUCs.txt'
file_model = './output/model/'+dataset+'model.pt'

TRAIN_BATCH_SIZE=512
TEST_BATCH_SIZE=512
cuda_name='cuda:0'

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
#train_dataset=torch.load('./data/processed/DUDE_SeqVec_molBERT_train_data_mol.pt')
#test_dataset=torch.load('./data/processed/DUDE_SeqVec_molBERT_test_data_mol.pt')


train_dataset=torch.load('./data/DUDE/data/processed/2graph_DUDETrain1_ProtTransAlbertBFDEmbedder_MolecularTransformer_train_data_mol.pt')
test_dataset=torch.load('./data/DUDE/data/processed/2graph_DUDETrain1_ProtTransAlbertBFDEmbedder_MolecularTransformer_test_data_mol.pt')

#train_dataset=torch.load('./data/DUDE/data/processed/'+dataset+'Train1_ProtTransAlbertBFDEmbedder_MolecularTransformer_train_data_mol.pt')
#test_dataset=torch.load('./data/DUDE/data/processed/'+dataset+'Train1_ProtTransAlbertBFDEmbedder_MolecularTransformer_test_data_mol.pt')

#train_dataset=torch.load('./data/DB/data/CTD/processed/DB_ProtTransAlbertBFDEmbedder_MolecularTransformer_train_data_mol.pt')
#valid_dataset=torch.load('./data/human/data/processed/human_ProtTransAlbertBFDEmbedder_MolecularTransformer_test_data_mol.pt')

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

#valid_loader=DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model=GCNNet().to(device)

lr=0.0001
trainer=Trainer(model,lr=lr)
tester=Tester(model)

FPER=np.array([])
TPER=np.array([])
human_fper=np.array([])
human_tper=np.array([])

bestAuc=0
vaild_AUC=0
for epoch in range(100):
    loss_train = trainer.train(train_loader,device,epoch+1)
    #AUC_test, precision_test, recall_test, RE1, RE2, RE3, RE4, fper, tper = tester.test(valid_loader, device)

    #if epoch>80 and AUC_test>vaild_AUC:
    #    vaild_AUC=AUC_test
    #    human_fper=fper
    #    human_tper=tper

    #AUCs = [epoch, float(loss_train),
    #        AUC_test, precision_test, recall_test, RE1, RE2, RE3, RE4]
    #print('\t'.join(map(str, AUCs)))
    AUC_test, precision_test, recall_test, RE1,RE2,RE3,RE4 ,fper, tper= tester.test(test_loader,device)


    if epoch>80 and AUC_test>bestAuc:
        bestAuc=AUC_test
        FPER=fper
        TPER=tper

    AUCs = [epoch, float(loss_train),
            AUC_test, precision_test, recall_test, RE1,RE2,RE3,RE4]
    tester.save_AUCs(AUCs, file_AUCs)
    tester.save_model(model, file_model)
    #para_AUCs="lr"+str(lr)+"_layernum128_dropout_org"
    #para_file_AUCs = './draw/param/' + para_AUCs + '_AUCs.txt'
    #tester.save_AUCs(AUCs, para_file_AUCs)
    print('\t'.join(map(str, AUCs)))

#AUC_test, precision_test, recall_test, RE1,RE2,RE3,RE4 ,human_fper, human_tper= tester.test(valid_loader,device)

#np.save('./draw/human_FPER.npy',human_fper)
#np.save('./draw/human_TPER.npy',human_tper)
#np.save('./draw/cele_TPER.npy',FPER)
#np.save('./draw/cele_FPER.npy',TPER)

plot_roc_curve(human_fper,human_tper)

