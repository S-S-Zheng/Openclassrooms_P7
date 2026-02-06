"""
Boucle d'entraînement, évaluation et génération de pseudo-labels.
"""

# imports
import torch
import torch.nn as nn
import numpy as np
# Affiche une barre de progression pendant l'entraînement.
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score





class Trainer:
    """
    Gère le cycle de vie de l'entraînement (Supervisé & Pseudo-labeling).
    """
    def __init__(self, model, device, criterion, optimizer):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = {'train_loss': [], 'val_f1': []}

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        self.history['train_loss'].append(epoch_loss)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        preds_all = []
        labels_all = []
        
        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.numpy())
                
        f1 = f1_score(labels_all, preds_all, average='weighted')
        self.history['val_f1'].append(f1)
        return f1, classification_report(labels_all, preds_all)

    def generate_pseudo_labels(self, unlabeled_loader, threshold=0.95):
        """
        Passe sur les données sans label et retourne celles où le modèle est confiant.
        Retourne : liste de chemins, liste de labels prédits
        """
        self.model.eval()
        pseudo_paths = []
        pseudo_labels = []
        
        print("Génération des pseudo-labels...")
        with torch.no_grad():
            for inputs, _, paths in tqdm(unlabeled_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                max_probs, preds = torch.max(probs, dim=1)
                
                # Filtrage par confiance
                mask = max_probs >= threshold
                
                if mask.any():
                    valid_paths = np.array(paths)[mask.cpu().numpy()]
                    valid_preds = preds[mask].cpu().numpy()
                    
                    pseudo_paths.extend(valid_paths)
                    pseudo_labels.extend(valid_preds)
                    
        return pseudo_paths, pseudo_labels