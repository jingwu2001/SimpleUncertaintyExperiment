import os
import torch
from viModel import BayesianMnistNet
from ensemble import DeterministicNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def saveBNNs(models, savedir) :
    
    for i, m in enumerate(models) :
        
        saveFileName = os.path.join(savedir, "BNN_model{}.pth".format(i))
        torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))

def saveEnsemble(models, savedir):
    for i, m in enumerate(models):
        saveFileName = os.path.join(savedir, f"Ensemble_model{i}.pth")
        torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))
    
def loadBNNs(savedir, device) :
    
    models = []

    if "mnist" in savedir:
        in_channels = 1
        input_size = (28, 28)
    elif "cifar10" in savedir:
        in_channels = 3
        input_size = (32, 32)
    
    for f in os.listdir(savedir) :
        if "BNN" not in f:
            continue
        model = BayesianMnistNet(in_channels=in_channels, input_size=input_size, p_mc_dropout=None)
        model.to(device)		
        model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
        models.append(model)
        
    return models

def loadEnsemble(savedir, device):
    models = []
    
    if "mnist" in savedir:
        in_channels = 1
        input_size = (28, 28)
    elif "cifar10" in savedir:
        in_channels = 3
        input_size = (32, 32)

    for f in os.listdir(savedir) :
        if "Ensemble" not in f:
            continue
        
        model =DeterministicNet(in_channels=in_channels, input_size=input_size, p_mc_dropout=None)
        model.to(device)		
        model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
        models.append(model)

    return models

def saveModels(bnn_models, ens_models, savedir):
    saveBNNs(bnn_models, savedir)
    saveEnsemble(ens_models, savedir)



# def saveModels(models, savedir) :
    
#     for i, m in enumerate(models) :
        
#         saveFileName = os.path.join(savedir, "model{}.pth".format(i))
        
#         torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))

# def saveEnsemble(models, savedir):
#     for i, m in enumerate(models):
#         saveFileName = os.path.join(savedir, f"ensemble_model{i}.pth")
#         torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))
    
# def loadModels(savedir) :
    
#     models = []
    
#     for f in os.listdir(savedir) :
        
#         model = BayesianMnistNet(p_mc_dropout=None)
#         model.to(device)		
#         model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
#         models.append(model)
        
#     return models

# def loadEnsemble(savedir):
#     models = []
    
#     for f in os.listdir(savedir) :
        
#         model =MnistNet(p_mc_dropout=None)
#         model.to(device)		
#         model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
#         models.append(model)
        
#     return models
