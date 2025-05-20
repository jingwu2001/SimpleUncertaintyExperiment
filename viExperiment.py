#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:04:42 2021

@author: laurent
"""

from dataset import getSets
from viModel import BayesianMnistNet
from ensemble import DeterministicNet
import utils

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


import os

import argparse as args
import sys

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader


if __name__ == "__main__" :
    
    parser = args.ArgumentParser(description='Train a BNN on Mnist')

    parser.add_argument('--dataset', default="mnist", help="Dataset to use. Options: mnist/cifar10")
    
    parser.add_argument('--filteredclass', type=int, default = 5, choices = [x for x in range(10)], help="The class to ignore during training")
    parser.add_argument('--testclass', type=int, default = 4, choices = [x for x in range(10)], help="The class to test against that is not the filtered class")
    
    parser.add_argument('--datadir', default = './data/', help="Direcctory where CIFAR10/MNIST data is downloaded")
    parser.add_argument('--savedir', default = "models", help="Directory where the models can be saved or loaded from")

    parser.add_argument('--notrain', action = "store_true", help="Load the models directly instead of training")
    parser.add_argument('--notrainBNN', action = "store_true", help="Load the BNN models directly instead of training")
    parser.add_argument('--notrainEnsemble', action = "store_true", help="Load the ensemble models directly instead of training")
    
    parser.add_argument('--nepochs', type=int, default = 10, help="The number of epochs to train for")
    parser.add_argument('--nbatch', type=int, default = 64, help="Batch size used for training")
    parser.add_argument('--numworkers', type=int, default=2, help="num_workers in DataLoader")
    parser.add_argument('--nruntests', type=int, default = 50, help="The number of pass to use at test time for monte-carlo uncertainty estimation")
    parser.add_argument('--learningrate', type=float, default = 5e-3, help="The learning rate of the optimizer")

    parser.add_argument('--numnetworksBNN', type=int, default = 10, help="The number of BNN networks to train to make an ensemble")
    parser.add_argument('--numnetworksEnsemble', type=int, default=5, help="The number of networks in the deterministic ensemble")

    parser.add_argument('--trainonly', action = "store_true", help="train the models and stop. No inference would be done if this is specified")

    parser.add_argument('--advancedmetrics', action="store_true", help="Calculate advanced metrics")
    parser.add_argument('--runtests', action="store_true", help='do inference on the test set')
    parser.add_argument('--lrSGD', type=float, default=0.1, help="learning rate for SGD")

    parser.add_argument('--checkpsteps', type=int, default=10, help="how many epochs between checkpoint savings")

    
    args = parser.parse_args()
    plt.rcParams["font.family"] = "serif"


    ## put identify onto gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ensure output folder exists
    os.makedirs(args.savedir, exist_ok=True)
    img_dir = args.savedir + '/' + "plots"
    os.makedirs(img_dir, exist_ok=True)
    task_dir = args.savedir + '/' + args.dataset
    os.makedirs(task_dir, exist_ok=True)
    model_dir = task_dir + '/' + "models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoints_dir = model_dir + "/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # sys.exit(0)

    plot_counter = 0
    def save_plot(name: str):
        """
        Saves the current figure to images/ as
        01_name.png, 02_name.png, etc.
        """
        global plot_counter  # if inside a function; otherwise use `global plot_counter`
        plot_counter += 1
        # fname = f"{plot_counter:02d}_{name}.png"
        fname = f"{name}.png"
        path = os.path.join(img_dir, fname)
        plt.savefig(path)
        print(f" saved {path}")


    
    train, test = getSets(args.dataset, datadir=args.datadir, filteredClass = args.filteredclass)
    train_filtered, test_filtered = getSets(args.dataset, datadir=args.datadir, filteredClass = args.filteredclass, removeFiltered = False)
    
    N = len(train)
    SEED = 1235                                   # pick any
    import random
    def seed_everything(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # if using CUDA

    ###############################################
    # 1.  FIRST RUN
    ###############################################
    seed_everything()

    
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.nbatch, num_workers=args.numworkers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.nbatch, num_workers=args.numworkers)
    
    # make sure we're using the right dataset
    test_data = next(iter(train_loader))
    print(f"length of training dataset: {N}, input shape: {test_data[0].shape}, label shape: {test_data[1].shape}")
    # sys.exit(0)
    batchLen = len(train_loader)
    digitsBatchLen = len(str(batchLen))
    
    bnn_models = []
    ens_models = []
    
    # Training or Loading
    if args.notrain :
        
        bnn_models = utils.loadBNNs(args.savedir)
        ens_models = utils.loadEnsemble(args.savedir)
        
    else :
        if args.dataset == "mnist":
            in_channels = 1
            input_size = (28, 28)

            
            for i in np.arange(args.numnetworksBNN) :
                print("Training model {}/{}:".format(i+1, args.numnetworksBNN))
                
                #Initialize the model
                model = BayesianMnistNet(in_channels=in_channels, input_size=input_size, p_mc_dropout=None) #p_mc_dropout=None will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
                model.to(device)
                loss = torch.nn.NLLLoss(reduction='mean') #negative log likelihood will be part of the ELBO
                
                optimizer = Adam(model.parameters(), lr=args.learningrate)
                optimizer.zero_grad()
                
                for n in np.arange(args.nepochs) :
                    
                    for batch_id, sampl in enumerate(train_loader) :
                        
                        images, labels = sampl
                        
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        pred = model(images, stochastic=True)
                        
                        logprob = loss(pred, labels)
                        l = N*logprob
                        
                        
                        modelloss = model.evalAllLosses()
                        l += modelloss
                        
                        optimizer.zero_grad()
                        l.backward()
                        
                        optimizer.step()
                        

                        print("\r", ("\tEpoch {}/{}: Train step {"+(":0{}d".format(digitsBatchLen))+"}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
                                                                                                        n+1, args.nepochs,
                                                                                                        batch_id+1,
                                                                                                        batchLen,
                                                                                                        torch.exp(-logprob.detach().cpu()).item(),
                                                                                                        modelloss.detach().cpu().item(),
                                                                                                        l.detach().cpu().item()), end="")
                print("")
                bnn_models.append(model)

            for i in np.arange(args.numnetworksEnsemble):
                print("Training model {}/{}:".format(i+1, args.numnetworksEnsemble))
                model = DeterministicNet(in_channels=in_channels, input_size=input_size, p_mc_dropout=None).to(device)
                loss_fn = torch.nn.CrossEntropyLoss() 
                optimizer = Adam(model.parameters(), lr=args.learningrate)
                optimizer.zero_grad()

                for n in np.arange(args.nepochs) :
                    for  batch_id, sampl in enumerate(train_loader):
                        images, labels = sampl
                        images = images.to(device)
                        labels = labels.to(device)
                        pred = model(images)

                        l = loss_fn(pred, labels)

                        optimizer.zero_grad()
                        l.backward()

                        optimizer.step()
                        print("\r", ("\tEpoch {}/{}: Train step {"+(":0{}d".format(digitsBatchLen))+"}/{} loss = {:.4f}          ").format(
                                                                                    n+1, args.nepochs,
                                                                                    batch_id+1,
                                                                                    batchLen,
                                                                                    l.detach().cpu().item()), end="")
                print()

                ens_models.append(model)

        elif args.dataset == "cifar10":

            ###### prepare a validation dataset
            if hasattr(train, "targets"):
                print("training dataset is not a subset")
                labels = np.array(train.targets)      # e.g. shape (N,)
            else:
                print("dataset is a subset")
                # train_ds is a Subset, so grab labels via .dataset and .indices
                base_labels = np.array(train.dataset.targets)
                labels     = base_labels[train.indices]

            # 3) Create the splitter
            val_frac = 0.1
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_frac,
                random_state=42
            )
            # sys.exit(0)

            # 4) Generate train/val indices
            train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

            # 5) Wrap in Subsets
            train_split = Subset(train, train_idx)
            val_split   = Subset(train, val_idx)
            n_train = len(train_split)
            print(len(train_split), len(val_split))
            # sys.exit(0)
            # 6) Build your loaders
            train_loader = DataLoader(train_split, batch_size=args.nbatch, shuffle=True, num_workers=args.numworkers)
            val_loader   = DataLoader(val_split,   batch_size=args.nbatch, shuffle=False, num_workers=args.numworkers)

            #####
            # val_loader_test = DataLoader(val_split, batch_size=len(val_split))
            # labels = next(iter(val_loader_test))[1]
            # counts = torch.bincount(labels)
            # print(f"label counts: {counts}")  
            ######


            print(f"Train on {len(train_split)} samples, validate on {len(val_split)} samples")

            in_channels = 3
            input_size = (32, 32)
            from torch.optim.lr_scheduler import StepLR

            n_models = args.numnetworksBNN
            n_epochs = args.nepochs

            all_train_nll = []
            all_train_kl  = []
            all_train_acc = []
            all_val_nll   = []
            all_val_acc   = []

            if not args.notrainBNN:
                for i in range(args.numnetworksBNN):
                    train_nll_hist = []
                    train_kl_hist  = []
                    train_acc_hist = []
                    val_nll_hist   = []
                    val_acc_hist   = []
                    print(f"Training BNN model {i+1}/{args.numnetworksBNN}")
                    
                    # 1) initialize
                    model     = BayesianMnistNet(in_channels, input_size, p_mc_dropout=None)
                    model.to(device)
                    n_train   = len(train_loader.dataset)
                    loss_fn   = torch.nn.NLLLoss(reduction='mean')
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate)
                    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
                    
                    # 2) per-epoch tracking
                    for epoch in range(1, args.nepochs+1):
                        model.train()
                        running_nll = 0.0
                        running_kl  = 0.0
                        running_correct = 0
                        
                        for batch_id, (images, labels) in enumerate(train_loader, 1):
                            images, labels = images.to(device), labels.to(device)
                            
                            # forward: stochastic=True samples weights once per batch
                            pred = model(images, stochastic=True)
                            logprob = loss_fn(pred, labels)
                            l = (n_train / images.size(0)) * logprob
                            # l = N*logprob

                            modelloss = model.evalAllLosses()
                            l += modelloss
                            optimizer.zero_grad()
                            l.backward()
                            optimizer.step()

                            
                            # accumulate for reporting
                            running_nll    += logprob.item()  * images.size(0)
                            running_kl     += modelloss.item()
                            preds           = pred.argmax(dim=1)
                            running_correct += (preds == labels).sum().item()
                            
                        # scheduler step & epoch metrics
                        scheduler.step()
                        avg_nll = running_nll / n_train
                        avg_kl  = running_kl  / len(train_loader)
                        acc     = running_correct / n_train

                        train_nll_hist.append(avg_nll)
                        train_kl_hist .append(avg_kl)
                        train_acc_hist.append(acc)

                        # Validation
                        model.eval()
                        val_nll = val_correct = 0
                        with torch.no_grad():
                            for images, labels in val_loader:
                                lp = model(images.to(device), stochastic=True)
                                l  = loss_fn(lp, labels.to(device))
                                val_nll     += l.item() * images.size(0)
                                val_correct += (lp.argmax(1)==labels.to(device)).sum().item()
                        val_nll /= len(val_loader.dataset)
                        val_acc  = val_correct / len(val_loader.dataset)
                        val_nll_hist.append(val_nll)
                        val_acc_hist.append(val_acc)

                        if epoch % args.checkpsteps == 0:
                            saveFileName = os.path.join(checkpoints_dir, f"BNN_model_{i}_epoch{epoch:03d}.pth")
                            torch.save({"model_state_dict": model.state_dict()}, os.path.abspath(saveFileName))
                            torch.save({
                                "epoch":   epoch+1,
                                "opt":     optimizer.state_dict(),
                                "sched":   scheduler.state_dict(),
                                "rng_cpu": torch.get_rng_state(),
                                "rng_cuda": torch.cuda.get_rng_state_all(),
                            }, f"{checkpoints_dir}/bnn_ckpt{i}_{epoch}.pt")


                            print(f"BNN Model {i}, Epoch {epoch}: train NLL={avg_nll:.4f}, acc={acc:.4f} │ val NLL={val_nll:.4f}, acc={val_acc:.4f}")

                    # — after all epochs ——
                    all_train_nll.append(train_nll_hist)
                    all_train_kl .append(train_kl_hist)
                    all_train_acc.append(train_acc_hist)
                    all_val_nll  .append(val_nll_hist)
                    all_val_acc  .append(val_acc_hist)

                    # 3) after training, switch to eval
                    model.eval()
                    bnn_models.append(model)

                train_nll_arr = np.array(all_train_nll)
                train_kl_arr  = np.array(all_train_kl)
                train_acc_arr = np.array(all_train_acc)
                val_nll_arr   = np.array(all_val_nll)
                val_acc_arr   = np.array(all_val_acc)


                metrics = {
                    "train_nll": train_nll_arr,   # (n_models, n_epochs)
                    "train_kl" : train_kl_arr,
                    "train_acc": train_acc_arr,
                    "val_nll"  : val_nll_arr,
                    "val_acc"  : val_acc_arr,
                }

                epochs = np.arange(1, metrics["train_nll"].shape[1] + 1)

                # ------------------------------------------------------------------
                # 2)   make one figure per metric ----------------------------------
                # ------------------------------------------------------------------
                for name, arr in metrics.items():
                    plt.figure(figsize=(6, 4))

                    # plot each model in light colour
                    for row in arr:
                        plt.plot(epochs, row, alpha=0.25, linewidth=1)

                    # mean ± std envelope
                    mean = arr.mean(axis=0)
                    std  = arr.std(axis=0)
                    plt.fill_between(epochs, mean - std, mean + std, color="C0", alpha=0.2)
                    plt.plot(epochs, mean, color="C0", linewidth=2, label="mean ± std")

                    plt.xlabel("Epoch")
                    ylabel = "Accuracy" if "acc" in name else ("KL loss" if "kl" in name else "NLL loss")
                    plt.ylabel(ylabel)
                    plt.title(name.replace("_", " ").title())
                    plt.legend(frameon=False)
                    plt.grid(alpha=0.3)

                    save_plot("bnn_" + name)        # <- your helper
                    plt.close()

                if args.savedir is not None :
                    utils.saveBNNs(models=bnn_models, savedir=model_dir)


            if not args.notrainEnsemble:
                all_train_loss = []
                all_train_acc = []
                all_val_loss   = []
                all_val_acc   = []


                for i in np.arange(args.numnetworksEnsemble):
                    train_loss_hist = []
                    train_acc_hist  = []
                    val_loss_hist   = []
                    val_acc_hist   = []
                    print("Training model {}/{}:".format(i+1, args.numnetworksEnsemble))
                    model = DeterministicNet(in_channels=in_channels, input_size=input_size, p_mc_dropout=None).to(device)
                    loss_fn = torch.nn.CrossEntropyLoss() 
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
                    )
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

                    # (2) Training + validation
                    for epoch in range(1, args.nepochs+1):
                        # — train —
                        model.train()
                        tloss = tcorrect = 0
                        for x, y in train_loader:
                            x,y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            out = model(x)
                            loss = loss_fn(out, y)
                            loss.backward()
                            optimizer.step()

                            tloss    += loss.item() * x.size(0)
                            tcorrect += (out.argmax(1)==y).sum().item()

                        tloss /= len(train_loader.dataset)
                        tacc  = tcorrect / len(train_loader.dataset)

                        # — validate —
                        model.eval()
                        vloss = vcorrect = 0
                        with torch.no_grad():
                            for x, y in val_loader:
                                x,y = x.to(device), y.to(device)
                                out = model(x)
                                l   = loss_fn(out, y)
                                vloss    += l.item() * x.size(0)
                                vcorrect += (out.argmax(1)==y).sum().item()
                        vloss /= len(val_loader.dataset)
                        vacc  = vcorrect / len(val_loader.dataset)

                        # — step scheduler & log —
                        scheduler.step()
                        lr = scheduler.get_last_lr()[0]

                        

                        train_loss_hist.append(tloss)
                        train_acc_hist.append(tacc)
                        val_loss_hist.append(vloss)
                        val_acc_hist.append(vacc)

                        # Save models per checkpsteps epoch
                        if epoch % args.checkpsteps == 0:
                            saveFileName = os.path.join(checkpoints_dir, f"ENS_model_{i}_epoch{epoch:03d}.pth")
                            torch.save({"model_state_dict": model.state_dict()}, os.path.abspath(saveFileName))
                            torch.save({
                                "epoch":   epoch+1,
                                "opt":     optimizer.state_dict(),
                                "sched":   scheduler.state_dict(),
                                "rng_cpu": torch.get_rng_state(),
                                "rng_cuda": torch.cuda.get_rng_state_all(),
                            }, f"{checkpoints_dir}/ens_ckpt{epoch}.pt")
                            print(f"ENS Model {i}, Epoch {epoch}: train loss={tloss:.4f}, acc={tacc:.4f} │ val loss={vloss:.4f}, acc={vacc:.4f}")

                    
                    all_train_loss.append(train_loss_hist)
                    all_train_acc.append(train_acc_hist)
                    all_val_loss.append(val_loss_hist)
                    all_val_acc.append(val_acc_hist)
                    ens_models.append(model)

                # shapes  (n_models, n_epochs)
                train_loss_arr = np.asarray(all_train_loss)
                train_acc_arr  = np.asarray(all_train_acc)
                val_loss_arr   = np.asarray(all_val_loss)
                val_acc_arr    = np.asarray(all_val_acc)

                metrics = {
                    "train_loss": train_loss_arr,   # (n_models, n_epochs)
                    "train_acc" : train_acc_arr,
                    "val_loss"  : val_loss_arr,
                    "val_acc"   : val_acc_arr,
                }
                epochs = np.arange(1, metrics["train_nll"].shape[1] + 1)
                # ------------------------------------------------------------------
                # 2)  plot one figure per metric -----------------------------------
                # ------------------------------------------------------------------
                for name, arr in metrics.items():
                    plt.figure(figsize=(6, 4))

                    # draw each model (faint)
                    for row in arr:
                        plt.plot(epochs, row, alpha=0.25, linewidth=1)

                    # mean ± std
                    mean = arr.mean(0)
                    std  = arr.std(0)
                    plt.fill_between(epochs, mean - std, mean + std,
                                    color="C0", alpha=0.2)
                    plt.plot(epochs, mean, color="C0", linewidth=2,
                            label="mean ± std")

                    plt.xlabel("Epoch")
                    ylabel = "Accuracy" if "acc" in name else "Loss"
                    plt.ylabel(ylabel)
                    plt.title(name.replace("_", " ").title())
                    plt.grid(alpha=0.3)
                    plt.legend(frameon=False)

                    save_plot(name)
                    plt.close()

                # Save models
                if args.savedir is not None :
                    utils.saveEnsemble(models=ens_models, savedir=model_dir)

    
    if args.savedir is not None :
        utils.saveModels(bnn_models=bnn_models, ens_models=ens_models, savedir=model_dir)
    
    print("FINISHED TRAINING WITHOUT ERRORS")
    
    if args.trainonly:
        sys.exit(0)


    if args.advancedmetrics:
        pred_prob = {}

        if args.runtests:
            train_unfiltered, test_unfiltered = getSets(datadir=args.datadir, filteredClass=None) # all data
            assert len(test_unfiltered) == 10000
            # test_unfiltered_loader = DataLoader(test_unfiltered, batch_size=len(test_unfiltered))
            # images, labels = next(iter(test_unfiltered_loader))
            # images = images.to(device)
            # labels = labels.to(device)
            # # BNN
            # pred_prob['bnn'] = torch.zeros((args.nruntests, len(test_unfiltered), 10))
            # for i in np.arange(args.nruntests) :
            #     # print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
            #     model = np.random.randint(args.numnetworksBNN)
            #     model = models[model]
            #     output = torch.exp(model(images))
            #     pred_prob["bnn"][i, :, :] = output
            #     print(f"bnn: {i + 1}/{args.nruntests} models inferenced")
            
            # # Ensemble
            # pred_prob['ens'] = torch.zeros((len(ens_models), len(test_unfiltered), 10))

            # for i, m in enumerate(ens_models):
            #     output = torch.nn.functional.softmax(m(images), dim=-1)
            #     pred_prob["ens"][i, :, :] = output
            #     print(f"ensemble: {i + 1}/{len(ens_models)} models inferenced")

            # print(f'pred_prob["bnn"].shape: {pred_prob["bnn"].shape}')
            # print(f'pred_prob["ens"].shape: {pred_prob["ens"].shape}')
            # print(images.shape)
            # print(labels.shape)



            test_loader = DataLoader(test_unfiltered,
                         batch_size=256,   # <<-- key change
                         shuffle=False,
                         num_workers=2,
                         pin_memory=True)
            all_labels = []
            all_images = []
            with torch.no_grad():
                for img, lbl in test_loader:          # imgs are ignored with “_”
                    all_images.append(img.cpu())
                    all_labels.append(lbl.cpu())    # keep on CPU
            images = torch.cat(all_images, dim=0)
            labels = torch.cat(all_labels, dim=0)   # (N,)  where N = len(test_unfiltered)
            print("label tensor shape:", labels.shape)

            def eval_on_loader(net, loader, device, transform):
                """
                Run `net` on `loader`, apply `transform` to its output (e.g. exp or softmax),
                and return a stacked (N, 10) tensor of probabilities on the CPU.
                """
                net.eval()
                probs = []
                with torch.no_grad():
                    for imgs, _ in loader:          # small mini-batches ⇒ low VRAM
                        imgs = imgs.to(device, non_blocking=True)
                        out  = net(imgs)            # raw logits *or* log-probs
                        probs.append(transform(out).cpu())
                        del out, imgs               # keep GPU memory clean
                return torch.cat(probs, dim=0)      # (N, 10) on CPU
            
            N = len(test_unfiltered)
            pred_prob = {}

            # ------------ BNN draws -------------------------------------------
            pred_prob['bnn'] = torch.zeros(args.nruntests, N, 10)
            for i in range(args.nruntests):
                net = bnn_models[np.random.randint(args.numnetworksBNN)]
                probs = eval_on_loader(net, test_loader, device, transform=torch.exp)  # log-probs → probs
                pred_prob['bnn'][i] = probs
                print(f"bnn: {i+1}/{args.nruntests} models inferenced")

            # ------------ deterministic ensemble -------------------------------
            pred_prob['ens'] = torch.zeros(len(ens_models), N, 10)
            for i, m in enumerate(ens_models):
                probs = eval_on_loader(
                    m, test_loader, device,
                    transform=lambda x: torch.nn.functional.softmax(x, dim=-1)          # logits → probs
                )
                pred_prob['ens'][i] = probs
                print(f"ensemble: {i+1}/{len(ens_models)} models inferenced")



            print(images.shape)
            print(labels.shape)
            print(pred_prob["bnn"].shape)
            print(pred_prob["ens"].shape)
            # Save the test results
            torch.save(images, os.path.join(args.testoutputdir, "images.pt"))
            torch.save(labels, os.path.join(args.testoutputdir, "labels.pt"))
            torch.save(pred_prob["bnn"], os.path.join(args.testoutputdir, "bnn_pred_prob.pt"))
            torch.save(pred_prob["ens"], os.path.join(args.testoutputdir, "ens_pred_prob.pt"))
        else:
            images = torch.load(os.path.join(args.testoutputdir, "images.pt"))
            labels = torch.load(os.path.join(args.testoutputdir, "labels.pt"))
            pred_prob['bnn'] = torch.load(os.path.join(args.testoutputdir, "bnn_pred_prob.pt"))
            pred_prob['ens'] = torch.load(os.path.join(args.testoutputdir, "ens_pred_prob.pt"))
            print(images.shape)
            print(labels.shape)
            print(pred_prob["bnn"].shape)
            print(pred_prob["ens"].shape)


        # now pred_prob contains the predicted probabilities for the entire test set

        # deal with ensemble probabilties
        # In Simple and Scalable Predictive Uncertainty the predicted probability is just the mean across members
        images = torch.as_tensor(images).cpu()
        labels = torch.as_tensor(labels).cpu()
        pred_prob["bnn"] = torch.as_tensor(pred_prob["bnn"]).cpu()
        pred_prob["ens"] = torch.as_tensor(pred_prob["ens"]).cpu()


        #### uncertainty aggregators
        def one_minus_max_pibar(prob):
            pibar = torch.mean(prob, dim=0) # the average of predicted probabilities
            return 1 - torch.max(pibar, dim=-1).values
        # print(one_minus_max_pibar(pred_prob['ens']).shape)

        def varf(probs, unbiased=True):
            """
            Row-wise average of column variance

            Args
            ----
            probs     : Tensor shape (N, C)
            unbiased  : if True, use 1/(C-1) (sample variance);
                        if False, use 1/C      (population variance).

            Returns
            -------
            Tensor shape (N,)  - variance for each row.
            """
            var_for_classes = probs.var(dim=0, unbiased=unbiased)
            return torch.mean(var_for_classes, dim=-1)

        def row_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
            """
            Compute H(p) = −Σ_c p_c log p_c  for each row of a (M, C) probability matrix.

            Parameters
            ----------
            probs : torch.Tensor
                Shape (M, C); each row must already sum to 1.
            eps   : float
                Small number added to avoid log(0).  1e-12 is usually safe for float32.

            Returns
            -------
            torch.Tensor
                Shape (M,)  - entropy for every row (natural-log units, i.e. nats).
            """
            p = probs.clamp_min(eps)          # ensure strictly > 0 to avoid log(0)
            return -(p * p.log()).sum(dim=1)  # (M,)
        

        def unc_one_minus_max_pibar(p):           # p → (M, N, C)
            return one_minus_max_pibar(p)

        def unc_varf(p):
            return varf(p.mean(dim=0), unbiased=False)

        def unc_entropy(p):
            return row_entropy(p.mean(dim=0))
        
        aggregators = {
            "1 - max prob":   unc_one_minus_max_pibar,
            "Variance":    unc_varf,
            "Entropy": unc_entropy,
        }

        pibar_bnn = pred_prob["bnn"].mean(dim=0)                 # (N, C)
        pibar_ens = pred_prob["ens"].mean(dim=0)

        preds = {
            "bnn": torch.argmax(pibar_bnn, dim=1),               # (N,)
            "ens": torch.argmax(pibar_ens, dim=1),
        }

        probs = {
            "bnn": pred_prob["bnn"],                             # (M_bnn, N, C)
            "ens": pred_prob["ens"],                             # (M_ens, N, C)
        }

        print(f"BNN accuracy: {(preds['bnn'] == labels).float().mean()}")
        print(f"ENS accuracy: {(preds['ens'] == labels).float().mean()}")

        def ap(labels, predictions, uncertainty, n_intervals):
            """
            Abstained prediction with uncertainty 
            LABELS        :(N,): label
            PREDICTIONS   :(N,), the prediction of a model
            UNCERTAINTY   :(N,)  uncertainty for each sample 

            return value: (n_intervals, ), the accuracy for each level
            """
    
            # N = labels.size(0)
            # correct_true = (labels == predictions)
    
            # # sort uncertainties to pick quantile thresholds
            # sorted_u, _ = torch.sort(uncertainty)   # shape (N,)
            
            # accuracies = torch.zeros(n_intervals, dtype=torch.float32)

            # for i in range(n_intervals):
            #     # coverage fraction
            #     k = max(int((i + 1) / n_intervals * N), 1)  # at least one sample
            #     thresh  = sorted_u[k - 1]                   # kth smallest uncertainty
            #     kept    = uncertainty <= thresh             # Boolean mask
            #     accuracies[i] = correct_true[kept].float().mean()
            # return accuracies
            N = labels.numel()
            correct_true = (labels == predictions)

            # ---- sort once, keep permutation --------------------------------------------------
            sorted_u, perm = torch.sort(uncertainty)
            correct_sorted = correct_true[perm]

            cov_acc = torch.empty(n_intervals, dtype=torch.float32,
                                device=labels.device)

            # ---- iterate over coverage levels -------------------------------------------------
            for i in range(n_intervals):
                keep_cnt = max((i + 1) * N // n_intervals, 1)   # make sure ≥1 sample
                cov_acc[i] = correct_sorted[:keep_cnt].float().mean()

            return cov_acc
        
        def cp(labels, predictions, uncertainty, n_intervals: int):
            """
            ROC curve for using *uncertainty* to predict correctness.

            Positive class  =  sample is correct  (labels == predictions)
            Score           =  −uncertainty       (larger ⇒ more likely correct)

            Returns
            -------
            tpr_curve : Tensor, shape (n_intervals,)
                True-positive rate at FPR = 1/n_intervals, 2/n_intervals, …, 1.
                Pass this straight into `area_under_curve(tpr_curve, n_intervals)`
                to approximate AUROC.
            """
            # ------------------------------------------------------------------
            # 1) Build Boolean labels and scores
            # ------------------------------------------------------------------
            pos_mask = (labels == predictions)          # positives = correct samples
            scores   = -uncertainty                     # higher ⇒ more likely correct

            # ------------------------------------------------------------------
            # 2) Stable sort by score (descending)
            # ------------------------------------------------------------------
            order = torch.argsort(scores,
                                descending=True,)
            pos_sorted = pos_mask[order].float()        # (N,)

            # cumulative counts as threshold moves down
            cum_tp = torch.cumsum(pos_sorted, dim=0)                # (N,)
            cum_fp = torch.cumsum(1.0 - pos_sorted, dim=0)          # (N,)

            n_pos = cum_tp[-1].item()
            n_neg = cum_fp[-1].item()

            # Handle edge cases (all pos / all neg)
            if n_pos == 0 or n_neg == 0:
                # degenerate ROC: either (0,0)→(0,1) or (0,0)→(1,0)
                tpr_curve = torch.zeros(n_intervals, dtype=torch.float32,
                                        device=labels.device)
                if n_pos:      # all positives → TPR is always 1
                    tpr_curve.fill_(1.0)
                return tpr_curve

            # ------------------------------------------------------------------
            # 3) Sample TPR at uniformly spaced FPR grid points
            # ------------------------------------------------------------------
            # Desired FP counts: 1/n_int, 2/n_int, …, n_neg
            fp_targets = torch.linspace(1, n_neg, steps=n_intervals,
                                        dtype=torch.float32, device=labels.device)

            # searchsorted returns the FIRST index where cum_fp >= target
            idx = torch.searchsorted(cum_fp, fp_targets)

            # ensure idx stays within range
            idx = torch.clamp(idx, max=len(cum_fp) - 1)

            tpr_curve = cum_tp[idx] / n_pos            # Tensor (n_intervals,)

            return tpr_curve

        # def cp(labels, predictions, uncertainty, n_intervals):
        #     """
        #     Correctness prediction with uncertainty 
        #     LABELS        :(N,): label
        #     PREDICTIONS   :(N,), the prediction of a model
        #     UNCERTAINTY   :(N,)  uncertainty for each sample 

        #     return value: (n_intervals, ), the accuracy for each level
        #     """
        #     # N = labels.size(0)
        #     # correct_true = (labels == predictions)
    
        #     # # sort uncertainties to pick quantile thresholds
        #     # sorted_u, _ = torch.sort(uncertainty)   # shape (N,)
            
        #     # meta_accuracies = torch.zeros(n_intervals, dtype=torch.float32)
        #     # for i in range(n_intervals):
        #     #     # determine quantile index for (i+1)/n_intervals
        #     #     frac = (i + 1) / n_intervals
        #     #     idx  = min(int(frac * N), N - 1)
        #     #     thresh = sorted_u[idx]

        #     #     # predict “correct” if u <= thresh
        #     #     pred_correct = (uncertainty <= thresh)

        #     #     # compare to true correctness
        #     #     meta_accuracies[i] = (pred_correct == correct_true).float().mean()
            
        #     # return meta_accuracies
        #     N = labels.numel()
        #     correct_true = (labels == predictions)                       # ground truth

        #     # ---- sort once, keep permutation --------------------------------------------------
        #     sorted_u, perm = torch.sort(uncertainty)
        #     correct_sorted = correct_true[perm]                          # align with sort

        #     acc = torch.empty(n_intervals, dtype=torch.float32,
        #                     device=labels.device)

        #     # ---- iterate over coverage levels -------------------------------------------------
        #     for i in range(n_intervals):
        #         keep_cnt = (i + 1) * N // n_intervals        # #points predicted “correct”
        #         pred_correct = torch.zeros(N, dtype=torch.bool, device=labels.device)
        #         pred_correct[:keep_cnt] = True               # first keep_cnt indices → True

        #         acc[i] = (pred_correct == correct_sorted).float().mean()

        #     return acc
        def ood_detect(labels,
                    predictions,        # <- kept for API compatibility, not used
                    uncertainty,
                    n_intervals: int,
                    ood_class: int = 5):
            """
            ROC curve for detecting OOD samples with *uncertainty*.

            Positive class  =  labels == ood_class
            Score           =  uncertainty         (larger ⇒ more likely OOD)

            Returns
            -------
            tpr_curve : Tensor, shape (n_intervals,)
                True-positive rate at FPR = 1/n, 2/n, …, 1.
                Feeding it to `area_under_curve(tpr_curve, n_intervals)` gives AUROC.
            """
            # ------------------------------------------------------------------
            # 1) Boolean labels (positives = OOD) and scores
            # ------------------------------------------------------------------
            pos_mask = (labels == ood_class)
            scores   = uncertainty                    # ↑  uncertainty ⇒ ↑  OOD

            # ------------------------------------------------------------------
            # 2) Stable sort by score (descending)
            # ------------------------------------------------------------------
            order = torch.argsort(scores,
                                descending=True,)
            pos_sorted = pos_mask[order].float()      # (N,)

            cum_tp = torch.cumsum(pos_sorted, dim=0)         # cumulative TPs
            cum_fp = torch.cumsum(1.0 - pos_sorted, dim=0)   # cumulative FPs

            n_pos = cum_tp[-1].item()
            n_neg = cum_fp[-1].item()

            # Edge cases: all pos or all neg
            if n_pos == 0 or n_neg == 0:
                tpr_curve = torch.zeros(n_intervals, dtype=torch.float32,
                                        device=labels.device)
                if n_pos:                                # all positives
                    tpr_curve.fill_(1.0)
                return tpr_curve

            # ------------------------------------------------------------------
            # 3) Sample TPR at uniformly spaced FPR grid points
            # ------------------------------------------------------------------
            # Desired FP counts: 1, 2, …, n_neg  split into n_intervals steps
            fp_targets = torch.linspace(1, n_neg, steps=n_intervals,
                                        dtype=torch.float32, device=labels.device)

            # For each FP target, find first idx with cum_fp >= target
            idx = torch.searchsorted(cum_fp, fp_targets)
            idx = torch.clamp(idx, max=len(cum_fp) - 1)

            tpr_curve = cum_tp[idx] / n_pos           # Tensor (n_intervals,)

            return tpr_curve

        # def ood_detect(labels, predictions, uncertainty, n_intervals, ood_class=5):
        #     """
        #     OOD detection accuracy at different uncertainty thresholds.
            
        #     For each i = 0..n_intervals-1:
        #     threshold = the ((i+1)/n_intervals) quantile of `uncertainty`
        #     pred_ood  = (uncertainty > threshold)
        #     accuracy  = mean(pred_ood == (labels == ood_class))
            
        #     Returns:
        #     Tensor of shape (n_intervals,) with detection accuracy at each level.
        #     """
        #     # # Ground‐truth mask for OOD
        #     # gt_ood = (labels == ood_class)        # BoolTensor of shape (N,)
            
        #     # # Sort uncertainties ascending
        #     # sorted_u, _ = torch.sort(uncertainty) # (N,)
        #     # N = labels.size(0)
            
        #     # acc = torch.zeros(n_intervals, dtype=torch.float32)
        #     # for i in range(n_intervals):
        #     #     frac = (i + 1) / n_intervals
        #     #     idx  = min(int(frac * N), N - 1)
        #     #     thresh = sorted_u[idx]
                
        #     #     # predict OOD if above threshold
        #     #     pred_ood = (uncertainty > thresh)
        #     #     if i == 0:
        #     #         print(f"{model_type}, {agg_name}, number of samples kept: {pred_ood.float().mean()}")
                
        #     #     # detection accuracy
        #     #     acc[i] = (pred_ood == gt_ood).float().mean()
            
        #     # return acc

        #         # 1) Sort once and keep the permutation ----------------------------------
        #     sorted_u, perm = torch.sort(uncertainty)      # (N,)
        #     labels_sorted = labels[perm]                               # (N,)
        #     # predictions_sorted = predictions[perm]  # <- handy if you need it later

        #     gt_ood = (labels_sorted == ood_class)                      # BoolTensor
        #     N      = labels_sorted.numel()
        #     acc    = torch.empty(n_intervals, dtype=torch.float32,
        #                         device=labels.device)

        #     # 2) Iterate over retention levels ---------------------------------------
        #     for i in range(n_intervals):
        #         # number of lowest-uncertainty samples to DROP
        #         drop_cnt = (i + 1) * N // n_intervals

        #         # predict OOD for everything *after* the drop_cnt index
        #         pred_ood = torch.zeros(N, dtype=torch.bool, device=labels.device)
        #         pred_ood[drop_cnt:] = True

        #         if i == 0:  # diagnostic print – adjust / remove if you like
        #             kept_frac = pred_ood.float().mean().item()
        #             print(f"{model_type}, {agg_name}, kept fraction: {kept_frac:.3f}")

        #         acc[i] = (pred_ood == gt_ood).float().mean()

        #     return acc

        def plot_curve(y, n_intervals, title="Curve", xlabel="Quantile", ylabel="Accuracy"):
            """
            Plot a curve given y-values at evenly spaced quantiles.
            
            Args:
            y (array-like): a sequence of length n_intervals
            n_intervals (int): number of points (length of y)
            title (str): plot title
            xlabel (str): label for the x-axis
            ylabel (str): label for the y-axis
            """
            y = np.asarray(y)
            x = np.arange(1, n_intervals + 1) / n_intervals
            plt.figure()
            plt.plot(x, y)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            if "Abstained" not in title:
                plt.ylim(0.0, 1.0) 
            # save_plot()

        def area_under_curve(y, n_intervals):
            """
            Compute the area under a curve using the trapezoidal rule.
            
            Args:
            y (array-like): a sequence of length n_intervals
            n_intervals (int): number of points (length of y)
            
            Returns:
            float: the estimated area under the curve
            """
            y = np.asarray(y)
            x = np.arange(1, n_intervals + 1) / n_intervals
            return np.trapz(y, x)

        # for prob in pred_probs["bnn"], pred_probs["ens"]
            # pred = torch.max(prob, dim=-1).values
            # for agg in one_minus_max_pibar, varf, row_entropy
                # for task in ood_detect, cp, ap
                    # unc = agg(prob)
                    # task(labels, pred, unc, n_intervals)
                    # plot_curve
                    # calculate area under curve and put it in a dictionary or something
            
        # for each task plot the area under curves for ((bnn/ens), agg)

        n_intervals = 100               # granularity of the curves
        auc = {task: {"bnn": {}, "ens": {}} for task in ["ap", "cp", "ood"]}

        for model_type in ["bnn", "ens"]:
            for agg_name, agg_fn in aggregators.items():

                # uncertainty vector  (N,)
                
                if agg_name == "Variance":
                    unc = varf(pred_prob[model_type])
                else:
                    unc = agg_fn(probs[model_type])

                if model_type == "ens" and agg_name == "1 - max prob":
                    print(unc[1], unc[100], unc[200], unc[1000])

                # run the three tasks ------------------------------------------------
                curve_ap  = ap (labels, preds[model_type], unc, n_intervals)
                curve_cp  = cp (labels, preds[model_type], unc, n_intervals)
                curve_ood = ood_detect(labels, preds[model_type], unc, n_intervals)

                # plot curves --------------------------------------------------------
                plot_curve(curve_ap,  n_intervals,
                        title=f"Abstained Prediction - {model_type} - {agg_name}",
                        ylabel="Accuracy")
                save_plot(f"ap - {model_type} - {agg_name}")

                plot_curve(curve_cp,  n_intervals,
                        title=f"Correctness Prediction - {model_type} - {agg_name}",
                        ylabel="Accuracy")
                save_plot(f"cp - {model_type} - {agg_name}")

                plot_curve(curve_ood, n_intervals,
                        title=f"OOD Detection - {model_type} - {agg_name}",
                        ylabel="Accuracy")
                save_plot(f"ood - {model_type} - {agg_name}")

                # store AUCs ----------------------------------------------------------
                auc["ap" ][model_type][agg_name] = area_under_curve(curve_ap,  n_intervals)
                auc["cp" ][model_type][agg_name] = area_under_curve(curve_cp,  n_intervals)
                auc["ood"][model_type][agg_name] = area_under_curve(curve_ood, n_intervals)

        # ------------------------------------------------------------------
        # 4.  the AUC summary dictionary
        # ------------------------------------------------------------------
        task_names = {"ap": "Abstained Prediction", "cp": "Correctness Prediction", "ood": "OOD Detection"}
        def plot_auc_bars(auc_dict):
            """
            Draw a bar plot for every task in `auc_dict`.

            Parameters
            ----------
            auc_dict : nested dict produced earlier
                auc[task][model_family][agg_name] = float
                e.g. task ∈ {'ap','cp','ood'}, model_family ∈ {'bnn','ens'}.
            """
            for task in sorted(auc_dict):

                # 1️⃣  gather (label, value) pairs
                pairs = []
                for fam in sorted(auc_dict[task]):            # bnn / ens
                    for agg in sorted(auc_dict[task][fam]):   # 1-max … Entropy …
                        pairs.append((f"{fam}-{agg}", auc_dict[task][fam][agg]))

                # 2️⃣  sort by value (descending)
                pairs.sort(key=lambda p: p[1], reverse=True)  # highest bar first

                # 3️⃣  unzip back into parallel lists
                labels, values = zip(*pairs)                  # tuples → lists

                # ---- plot ----------------------------------------------------------
                x = np.arange(len(labels))
                plt.figure(figsize=(1.2 * len(labels), 4))
                plt.bar(x, values, width=0.6)
                plt.xticks(x, labels, rotation=45, ha='right')
                plt.ylabel("Area under curve")
                plt.title(f"{task_names[task]} – AUC comparison")
                plt.tight_layout()
                save_plot(f"AOC_{task}_sorted")

        # ---------------------------------------------------------------------------
        # call the helper
        # ---------------------------------------------------------------------------
        plot_auc_bars(auc)
        sys.exit(0)