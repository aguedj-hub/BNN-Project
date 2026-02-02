#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:04:42 2021

@author: laurent
@ARTICLE{9756596,
author={Jospin, Laurent Valentin and Laga, Hamid and Boussaid, Farid and Buntine, Wray and Bennamoun, Mohammed},
journal={IEEE Computational Intelligence Magazine}, 
title={Hands-On Bayesian Neural Networks—A Tutorial for Deep Learning Users}, 
year={2022},
volume={17},
number={2},
pages={29-48},
doi={10.1109/MCI.2022.3155327}
}
"""

from dataset import getSets
from viModel import BayesianMnistNet

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import argparse as args
import random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"L'entraînement se fera sur : {device}")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def saveModels(models, savedir) :
    for i, m in enumerate(models) :
        saveFileName = os.path.join(savedir, "model{}.pth".format(i))
        torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))
    
def loadModels(savedir) :
    models = []
    for f in os.listdir(savedir) :
        model = BayesianMnistNet(p_mc_dropout=None)  
        model.to(device)   
        model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
        models.append(model)
    return models
  
def plot_weight_sparsity(model, title="Sparsité des poids"):
    """ Affiche l'histogramme des poids pour montrer l'effet du prior Laplace """
    weights = []
    for name, param in model.named_parameters():
        if 'weights_mean' in name:
            weights.extend(param.view(-1).detach().cpu().numpy())
    
    plt.figure(figsize=(10, 5))
    plt.hist(weights, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Valeur des poids")
    plt.ylabel("Fréquence")
    plt.yscale('log') # Log scale pour voir les petites valeurs proches de zéro
    plt.grid(True, which="both", ls="-", alpha=0.2)

def fgsm_attack(image, epsilon, data_grad):
    """ Génère une image adverse en ajoutant du bruit dans le sens du gradient """
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

if __name__ == "__main__" :
    
    def get_entropy(probs):
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    def plot_calibration(probs, labels, title, n_bins=10):
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = predictions.eq(labels)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []
        for i in range(n_bins):
            in_bin = confidences.gt(bin_boundaries[i]) & confidences.le(bin_boundaries[i+1])
            if in_bin.float().mean() > 0:
                bin_accs.append(accuracies[in_bin].float().mean().item())
                bin_confs.append(confidences[in_bin].mean().item())
        plt.figure(f"Calibration {title}")
        plt.title(f"Calibration Plot - {title}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.plot(bin_confs, bin_accs, "s-", label="Calibration")
        plt.xlabel("Confiance prédite"); plt.ylabel("Précision observée"); plt.legend()

    parser = args.ArgumentParser(description='Train a BNN on Mnist')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'laplace'])
    parser.add_argument('--filteredclass', type=int, default = 5, choices = [x for x in range(10)], help="The class to ignore during training")
    parser.add_argument('--testclass', type=int, default = 4, choices = [x for x in range(10)], help="The class to test against that is not the filtered class")
    parser.add_argument('--savedir', default = None, help="Directory where the models can be saved or loaded from")
    parser.add_argument('--notrain', action = "store_true", help="Load the models directly instead of training")
    parser.add_argument('--nepochs', type=int, default = 10, help="The number of epochs to train for")
    parser.add_argument('--nbatch', type=int, default = 64, help="Batch size used for training")
    parser.add_argument('--nruntests', type=int, default = 50, help="The number of pass to use at test time for monte-carlo uncertainty estimation")
    parser.add_argument('--learningrate', type=float, default = 5e-3, help="The learning rate of the optimizer")
    parser.add_argument('--numnetworks', type=int, default = 10, help="The number of networks to train to make an ensemble")
    
    args = parser.parse_args()
    plt.rcParams["font.family"] = "serif"
    
    train, test = getSets(filteredClass = args.filteredclass)
    train_filtered, test_filtered = getSets(filteredClass = args.filteredclass, removeFiltered = False)
    
    N = len(train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.nbatch)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.nbatch)
    
    batchLen = len(train_loader)
    digitsBatchLen = len(str(batchLen))
    
    models = []

    if args.notrain :
        models = loadModels(args.savedir)
    else :
        for i in np.arange(args.numnetworks) :
            print("Training model {}/{}:".format(i+1, args.numnetworks))
            model = BayesianMnistNet(prior_type = args.prior, p_mc_dropout=0.5)
            model.to(device)
            loss = torch.nn.NLLLoss(reduction='mean')
            optimizer = Adam(model.parameters(), lr=args.learningrate)
            optimizer.zero_grad()
            
            for n in np.arange(args.nepochs) :
                for batch_id, sampl in enumerate(train_loader) :
                    images, labels = sampl
                    images, labels = images.to(device), labels.to(device) 
                    pred = model(images, stochastic=True)
                    logprob = loss(pred, labels)
                    l = N*logprob
                    modelloss = model.evalAllLosses()
                    l += modelloss
                    optimizer.zero_grad(); l.backward(); optimizer.step()
                    print("\r", ("\tEpoch {}/{}: Train step {"+(":0{}d".format(digitsBatchLen))+"}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
                                                                                                    n+1, args.nepochs, batch_id+1, batchLen,
                                                                                                    torch.exp(-logprob.detach().cpu()).item(),
                                                                                                    modelloss.detach().cpu().item(),
                                                                                                    l.detach().cpu().item()), end="")
            print(""); models.append(model)
    
    if args.savedir is not None :
        saveModels(models, args.savedir)
    
    # --- Après l'entraînement, ajoute l'analyse de sparsité ---
    plot_weight_sparsity(models[0], title=f"Sparsité - Prior: {args.prior}")

    # --- Test d'Attaque Adversaire (Innovation) ---
    print("\nTesting Robustness (FGSM Attack):")
    epsilon = 0.2
    model = models[0]
    model.eval()

    # On récupère une image de test
    test_loader_single = DataLoader(test, batch_size=1)
    image, label = next(iter(test_loader_single))
    image, label = image.to(device), label.to(device)
    image.requires_grad = True

    # Calcul du gradient
    output = model(image, stochastic=True)
    loss = torch.nn.functional.nll_loss(output, label)
    model.zero_grad()
    loss.backward()

    # Création de l'image adverse
    perturbed_data = fgsm_attack(image, epsilon, image.grad.data)

    # Test de l'incertitude sur l'image corrompue
    with torch.no_grad():
        adv_samples = torch.zeros((args.nruntests, 10))
        for i in range(args.nruntests):
            adv_samples[i] = torch.exp(model(perturbed_data))
        
        mean_pred = adv_samples.mean(0)
        uncertainty = adv_samples.std(0).mean()
        print(f"Prédiction sur image corrompue: {mean_pred.argmax().item()}")
        print(f"Incertitude moyenne (Std): {uncertainty.item():.4f}")

    # Testing - SEEN CLASS
    if args.testclass != args.filteredclass :
        train_filtered_seen, test_filtered_seen = getSets(filteredClass = args.testclass, removeFiltered = False)
        print("\nTesting against seen class:")
        with torch.no_grad() :
            samples = torch.zeros((args.nruntests, len(test_filtered_seen), 10))
            test_loader = DataLoader(test_filtered_seen, batch_size=len(test_filtered_seen))
            images, labels = next(iter(test_loader))
            images = images.to(device)
            for i in np.arange(args.nruntests) :
                print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
                m_idx = np.random.randint(args.numnetworks)
                samples[i,:,:] = torch.exp(models[m_idx](images)).cpu()
                    
            print("")
            withinSampleMean = torch.mean(samples, dim=0)
            samplesMean = torch.mean(samples, dim=(0,1))
            withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
            acrossSamplesStd = torch.std(withinSampleMean, dim=0)
            
            entropies = get_entropy(withinSampleMean)
            print(f"\tEntropie moyenne: {entropies.mean():.4f}")
            plot_calibration(withinSampleMean, labels, "Seen Class")

            plt.figure("Seen class probabilities")
            plt.title("Seen Class Predictions")
            plt.bar(np.arange(10), samplesMean.numpy())
            plt.xlabel('digits'); plt.ylabel('digit prob'); plt.ylim([0,1]); plt.xticks(np.arange(10))
            
            plt.figure("Seen inner and outter sample std")
            plt.title("Seen Class Prediction Uncertainty")
            plt.bar(np.arange(10)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample (Aleatoric)")
            plt.bar(np.arange(10)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples (Epistemic)")
            plt.legend(); plt.xlabel('digits'); plt.ylabel('std digit prob'); plt.xticks(np.arange(10))
            
            network_probs_per_class_tensor = torch.zeros((args.numnetworks, 10))
            
            for j in range(args.numnetworks):
                preds = torch.exp(models[j](images)).cpu() 
                network_probs_per_class_tensor[j, :] = preds.mean(dim=0)
            
            plt.figure("Per-network mean predicted probabilities for seen class")
            plt.title("Per-network Mean Predicted Probabilities for Seen Class")
            for j in range(args.numnetworks):
                plt.plot(np.arange(10), network_probs_per_class_tensor[j, :], marker='o', label=f'Network {j+1}')
            plt.legend()
            

    # Testing - UNSEEN CLASS
    print("\nTesting against unseen class:")
    with torch.no_grad() :
        samples = torch.zeros((args.nruntests, len(test_filtered), 10))
        test_loader = DataLoader(test_filtered, batch_size=len(test_filtered))
        images, labels = next(iter(test_loader))
        images = images.to(device)
        for i in np.arange(args.nruntests) :
            print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
            m_idx = np.random.randint(args.numnetworks)
            samples[i,:,:] = torch.exp(models[m_idx](images)).cpu()
        print("")
        withinSampleMean = torch.mean(samples, dim=0)
        samplesMean = torch.mean(samples, dim=(0,1))
        withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
        acrossSamplesStd = torch.std(withinSampleMean, dim=0)
        
        entropies_unseen = get_entropy(withinSampleMean)
        print(f"\tEntropie moyenne (Incertitude): {entropies_unseen.mean():.4f}")

        plt.figure("Unseen class probabilities")
        plt.title("Unseen Class Predictions")
        plt.bar(np.arange(10), samplesMean.numpy())
        plt.xlabel('digits'); plt.ylabel('digit prob'); plt.ylim([0,1]); plt.xticks(np.arange(10))
        
        plt.figure("Unseen inner and outter sample std")
        plt.title("Unseen Class Prediction Uncertainty")
        plt.bar(np.arange(10)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample")
        plt.bar(np.arange(10)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples")
        plt.legend(); plt.xlabel('digits'); plt.ylabel('std digit prob'); plt.xticks(np.arange(10))
        
        network_probs_per_class_tensor = torch.zeros((args.numnetworks, 10))
        
        for j in range(args.numnetworks):
            preds = torch.exp(models[j](images)).cpu() 
            network_probs_per_class_tensor[j, :] = preds.mean(dim=0)
        
        plt.figure("Per-network mean predicted probabilities for unseen class")
        plt.title("Per-network Mean Predicted Probabilities for Unseen Class")
        for j in range(args.numnetworks):
            plt.plot(np.arange(10), network_probs_per_class_tensor[j, :], marker='o', label=f'Network {j+1}')
        plt.legend()

    # Testing - WHITE NOISE
    print("\nTesting against pure white noise:")
    with torch.no_grad() :
        l = 1000
        samples = torch.zeros((args.nruntests, l, 10))
        random_input = torch.rand((l,1,28,28)).to(device)
        for i in np.arange(args.nruntests) :
            print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
            m_idx = np.random.randint(args.numnetworks)
            samples[i,:,:] = torch.exp(models[m_idx](random_input)).cpu()
        print("")
        withinSampleMean = torch.mean(samples, dim=0)
        samplesMean = torch.mean(samples, dim=(0,1))
        withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
        acrossSamplesStd = torch.std(withinSampleMean, dim=0)
        
        entropies_noise = get_entropy(withinSampleMean)
        print(f"\tEntropie moyenne (Doute maximal attendu): {entropies_noise.mean():.4f}")

        plt.figure("White noise class probabilities")
        plt.title("White Noise Predictions")
        plt.bar(np.arange(10), samplesMean.numpy())
        plt.xlabel('digits'); plt.ylabel('digit prob'); plt.ylim([0,1]); plt.xticks(np.arange(10))
        
        plt.figure("White noise inner and outter sample std")
        plt.title("White Noise Prediction Uncertainty")
        plt.bar(np.arange(10)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample")
        plt.bar(np.arange(10)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples")
        plt.legend(); plt.xlabel('digits'); plt.ylabel('std digit prob'); plt.xticks(np.arange(10))
        
        network_probs_per_class_tensor = torch.zeros((args.numnetworks, 10))
        
        for j in range(args.numnetworks):
            preds = torch.exp(models[j](random_input)).cpu() 
            network_probs_per_class_tensor[j, :] = preds.mean(dim=0)
        
        plt.figure("Per-network mean predicted probabilities for noise class")
        plt.title("Per-network Mean Predicted Probabilities for Noise Class")
        for j in range(args.numnetworks):
            plt.plot(np.arange(10), network_probs_per_class_tensor[j, :], marker='o', label=f'Network {j+1}')
        plt.legend()
        
    plt.show()