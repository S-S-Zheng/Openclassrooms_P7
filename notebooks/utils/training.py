"""
Boucle d'entraînement, évaluation et génération de pseudo-labels.
"""

# imports
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import pandas as pd
# Affiche une barre de progression pendant l'entraînement.
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,recall_score,fbeta_score
)

from typing import Dict,List, Tuple,Any, Optional
from yaml.constructor import ConstructorError


class Trainer:
    """
    Gère le cycle de vie de l'entraînement, l'évaluation et la génération de pseudo-labels 
    pour un modèle de classification binaire.
    
    REMARQUE: les extensions Pytorch sont nombreuses ici donc petit rappel:
        - .item(): Extrait la valeur d'un tenseur ne contenant qu'un seul chiffre (ex: la loss)
        - .cpu(): Déplace les données GPU -> CPU OBLIGATOIRE pour évaluer/afficher/stocker
        - .numpy(): Transforme un tenseur en tableau, c'est le complément de cpu()
        - .flatten(): pour aplatir les dimensions
        - .tolist(): Transforme un tableau Numpy OU Tenseur en liste pthon standard pratique
    
    Attributes:
        model (nn.Module): Le réseau de neurones à entraîner/évaluer.
        device (torch.device): Le support de calcul (Cuda 'gpu' ou 'cpu').
        criterion (nn.modules.loss._Loss): La fonction de perte (ex: BCELoss).
        optimizer (torch.optim.Optimizer): L'algorithme d'optimisation (ex: Adam).
        threshold (float): Seuil de confiance pour valider un pseudo-label (0.0 à 1.0).
        history (Dict[str, List[float]]): Journal de bord stockant les métriques par époque.
    """
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        criterion: nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer,
        threshold: float = 0.95
    ):
        """
        Args:
            model (nn.Module): Le réseau de neurones à entraîner/évaluer.
            device (torch.device): Le support de calcul (Cuda 'gpu' ou 'cpu').
            criterion (nn.modules.loss._Loss): La fonction de perte (ex: BCELoss).
            optimizer (torch.optim.Optimizer): L'algorithme d'optimisation (ex: Adam).
            threshold (float): Seuil de confiance pour valider un pseudo-label (0.0 à 1.0).
            history (Dict[str, List[float]]): Journal de bord stockant les métriques par époque.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.threshold = threshold
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 
            'val_f2': [], 
            'val_precision': [], 
            'val_recall': [],
        }


    def train_epoch(self, dataloader:DataLoader)->float:
        """
        Exécute une passe complète (epoch) sur le jeu d'entraînement.
        
        Le modèle voit l'ensemble des images, calcule l'erreur et met à jour ses poids 
        via la rétropropagation du gradient.
        
        **C'est la partie supervisée. En tant que tel, cette méthode équivaut à de la 
        classification (comme catboost) mais gère des images/pixels 
        plutot que de la donnée tabulaires.**

        Args:
            dataloader (DataLoader): Flux de données d'entraînement (Images, Labels, Paths).
        
        Returns:
            float: La perte moyenne (Loss) calculée sur l'ensemble du dataset pour cette époque.
        """
        # Mode entraînement : active Dropout et BatchNorm
        self.model.train()
        # Accumulateur, la loss est la moyenne des batch multiplié par la taille du batch puis
        # ajouté au running_loss pour calculer a la fin de l'epoch, la moy tot du loss des images
        running_loss = 0.0
        
        for images, labels, _ in dataloader: # on déballe les infos
            # Transfert vers le device (GPU/CPU)
            images = images.to(self.device)
            # On prépare les labels au format attendu par BCELoss (Binary Cross Entropy) 
            # ==> attend des labels float et de même forme que l'output (Batch,1) 
            # d'où .float().view(-1,1) 
            labels = labels.to(self.device).float().view(-1,1) # similaire a reshape
            
            # ===== IMPORTANT: On efface les calculs du tour précédent ====
            # Réinitialisation des gradients (évite l'accumulation parasite)
            self.optimizer.zero_grad()
            
            # Le modèle fait son travail de prédiction (Propagation avant (Forward pass))
            outputs = self.model(images)
            
            # ===== IMPORTANT: On calcul l'erreur associé a chaque neurone =========
            # on compare la prédiction et la vérité ==> ecart faible label VS fort label
            loss = self.criterion(outputs, labels)
            # On compare la contrib des poids à la loss (retroprogation du gradient (Backward pass))
            loss.backward()
            
            # ===== IMPORTANT: On ajuste les poids du modèle pour la prochaine itération =====
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset) #type:ignore
        self.history['train_loss'].append(epoch_loss) # On stock la loss du train de cet epoch
        return epoch_loss


    def eval_metrics(
        self, 
        dataloader: DataLoader,
    )-> Dict[str,float]:
        """
        Évalue la performance du modèle sur un jeu de test/validation.\n
        Calcule le f2, la précision et le rappel en transformant les probabilités de 
        sortie en classes binaires (seuil 0.5).

        Args:
            dataloader(DataLoader): Flux de données de validation.
        
        Returns:
            Dict[str, float]: Dictionnaire contenant les scores {'f2', 'precision', 'recall'}.
        """
        # Determinisme, passe le modèle en mode évaluation (fige dropout et batchnorm par exemple)
        self.model.eval()
        probs_all = []
        preds_all = []
        labels_all = []
        
        # Gele le modèle car mémorisation inutile et pour économie mémoire
        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(self.device)
                
                # Le forward (prédiction du modèle) du modele
                y_proba = self.model(images)
                
                # 0.5 est le seuil de décision par défaut (le predict_proba)
                y_preds = (y_proba > 0.5).int().cpu().numpy()
                
                # Stockage pour calcul global des métriques sklearn
                probs_all.extend(y_proba.cpu().numpy().flatten().tolist())
                preds_all.extend(y_preds.flatten().tolist())
                labels_all.extend(labels.numpy().flatten().tolist())
                
        results = {
            'f2': float(fbeta_score(labels_all, preds_all, beta=2, zero_division=0)),
            'precision': float(precision_score(labels_all, preds_all, zero_division=0)),
            'recall': float(recall_score(labels_all, preds_all, zero_division=0)),
            'raw_data': {
                'probs': np.array(probs_all),
                'preds': np.array(preds_all),
                'labels': np.array(labels_all)
            } # Pour ECE et graphiques.
        }
        
        # MAJ de l'historique
        for key,value in results.items():
            self.history[f'val_{key}'].append(value) 
        return results


    def pseudo_labels(
        self, 
        unlabel_loader:DataLoader
    )->Tuple[List[str], List[int]]:
        """
        Génère des étiquettes pour les données inconnues via le mécanisme de confiance.\n
        Pour chaque image sans label, si le modèle prédit une classe avec une probabilité 
        supérieure ou égale au seuil de confiance, l'image est retenue (label faible).

        Args:
            unlabel_loader (DataLoader): Flux de données non-labellisées.
            
        Returns:
            Tuple[List[str], List[int]]: 
                - Liste des chemins de fichiers validés.
                - Liste des étiquettes (0 ou 1) attribuées par le modèle.
        """
        # Determinisme, passe le modèle en mode évaluation (fige dropout et batchnorm par exemple)
        self.model.eval()
        pseudo_paths = []
        pseudo_labels = []
        
        # Gele le modèle car mémorisation inutile et pour économie mémoire
        with torch.no_grad():
            for images, _, paths in tqdm(unlabel_loader, desc="Pseudo-labeling"):
                images = images.to(self.device)
                
                y_proba = self.model(images) # Sortie sigmoid
                
                # Pour le binaire, la confiance est soit proche de 1 (classe 1), 
                # soit proche de 0 (classe 0).
                # On calcule la distance par rapport à l'incertitude (0.5)
                confidence_1 = y_proba
                confidence_0 = 1 - y_proba
                
                # IMPORTANT: On réalise la comparaison (max_proba,y_pred,mask) sur GPU pour aller
                # plus vite et on repasse après en CPU
                max_proba, y_pred = torch.max(
                    torch.cat([confidence_0, confidence_1], dim=1),
                    dim=1
                )
                
                # Filtrage par confiance
                mask = max_proba >= self.threshold # Comparaison tenseur/float OK  (broadcasting)
                
                # Envoi vers CPU que pour l'indexation
                if mask.any():
                    # Filtrage des chemins (numpy est plus pratique ici pour le masque)
                    valid_paths = np.array(paths)[mask.cpu().numpy()].tolist()
                    valid_preds = y_pred[mask].cpu().numpy().tolist()
                    
                    pseudo_paths.extend(valid_paths)
                    pseudo_labels.extend(valid_preds)
                    
        return pseudo_paths, pseudo_labels
    
    
    def calculate_ece(
        self, 
        dataloader: Dict[str,np.ndarray]|DataLoader, 
        n_bins: int = 10
    ) -> float:
        """
        Calcule l'Expected Calibration Error qui mesure l'ecart entre la confiance du mdèle et
        sa précision réelle.\n
        ex: Si le modèle prédit 100 images avec une confiance de 0.99 et que moins de 99 images
        sont correctes alors le modèle est considéré trop sûr de lui (biais de confirmation).
        
        ==> C'est la somme pondérée en valeur absolue de l'ecart entre l'exactitude et la proba
        (si ece proche de 0, le modèle est bien calibré sinon c'est un menteur)
        
        Args:
            dataloader(DataLoader): Flux de données de validation. SOIT PARTIR DU DATALOADER
                SOIT DE L'OUTPUT DE EVAL_METRICS DANS RAW_DATAS POUR PAS REPETER
            n_bins(int): Regroupement des prédictions par intervalle de confiance. 
                ex: avec bin = 10 on créé 10 groupes sur l'intervalle (si ca va de 
                0 a 10 on aura [0-0.1] [0.1-0.2] ... [0.9-1.0])
            
        """
        self.model.eval()
        if isinstance(dataloader,dict):
            probs_all = dataloader['probs']
            preds_all = dataloader['labels']
            labels_all = dataloader['preds']
        else:
            probs_all = []
            preds_all = []
            labels_all = []

            # Gele le modèle car mémorisation inutile et pour économie mémoire
            with torch.no_grad():
                for images, labels, _ in dataloader:
                    images = images.to(self.device)
                    
                    # Le forward (prédiction du modèle) du modele
                    y_proba = self.model(images)
                    
                    # 0.5 est le seuil de décision par défaut (le predict_proba)
                    y_preds = (y_proba > 0.5).int().cpu().numpy()
                    
                    # Stockage pour calcul global des métriques sklearn
                    probs_all.extend(y_proba.cpu().numpy().flatten().tolist())
                    preds_all.extend(y_preds.flatten().tolist())
                    labels_all.extend(labels.numpy().flatten().tolist())
            
            # Tableaux numpy pour les calculs
            probs_all = np.array(probs_all)
            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)
        
        ece = 0.0
        # On crée nos intervalles (ex: 0.0-0.1, 0.1-0.2...)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for m in range(n_bins):
            # Masque pour l'intervalle (bin)
            bin_mask = (probs_all > bin_boundaries[m]) & (probs_all <= bin_boundaries[m+1])
            
            if np.any(bin_mask):
                # n_i / n : Proportion d'échantillons dans ce bin
                bin_weight = np.mean(bin_mask)
                # Accuracy du bin : moyenne des prédictions correctes
                bin_acc = np.mean(labels_all[bin_mask] == (preds_all[bin_mask] > 0.5))
                # Confiance moyenne du bin
                bin_conf = np.mean(probs_all[bin_mask])
                # Somme pondérée des écarts
                ece += bin_weight * np.abs(bin_acc - bin_conf)
                
        return float(ece)


# ================================================================================


class SslManager:
    """
    Gère la persistance des données d'une expérience de SSL.
    
    Sauvegarde des poids du modèle, log des métriques, archivage de la configuration 
    et des labels faibles.

    Attributes:
        root_path (Path): Dossier racine de l'expérience.
        ckpt_dir (Path): Sous-dossier pour les checkpoints (.pth, .ckpt).
        log_path (Path): Chemin vers le fichier CSV de suivi des métriques.
        config_path (Path): Chemin vers le fichier YAML des hyperparamètres.
        labels_path (Path): Chemin vers l'export des labels faibles (Parquet).
    """
    def __init__(
        self, 
        experiment_name:str="experiment_01",
        root_path:Path=Path.cwd(),
        extension_path:Path|str = "",
        
        
    ):
        """
        Initialise l'arborescence de l'expérience.
        
        Args:
            experiment_name: Nom unique du test.
            root_path: Dossier de base du projet.
            extension_path: Sous-dossier optionnel (ex: 'outputs/models').
        """
        self.root_path = root_path/Path(extension_path)/experiment_name
        self.root_path.mkdir(parents=True,exist_ok=True)
        
        self.ckpt_dir = self.root_path/"checkpoints"
        self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        
        self.log_path = self.root_path/"train_log.csv"
        self.config_path = self.root_path/"config.yaml"
        self.labels_path = self.root_path/"weak_labels.parquet"
        
        
    def save_config(self,config: Optional[Dict[str, Any]] = None,**kwargs:Any):
        """
        Sauvegarde la configuration au format yml
        
        Args:
            **kwargs: données de configuration
        """
        # On fusionne le dictionnaire 'config' et les 'kwargs'
        datasave = config.copy() if config is not None else {}
        datasave.update(kwargs)
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(datasave,f, default_flow_style=False)
    
    
    def log_metrics(self,metrics_dict: Optional[Dict[str, Any]] = None,**kwargs:Any):
        """
        Créé et/ou maj le fichier csv des métriques a chaque epoch
        """
        datasave = metrics_dict.copy() if metrics_dict is not None else {}
        datasave.update(kwargs)
        
        df = pd.DataFrame([datasave])
        header = not self.log_path.exists()
        df.to_csv(self.log_path, mode = 'a', index=False, header=header)

    
    def save_checkpoint(self, state: Dict[str, Any], is_best: bool = False):
        """
        Sauvegarde l'etat actuel et optionnellement le "best_model"
        
        Args:
            state(Dict[str, Any]): Dictionnaire contenant 'state_dict', 'optimizer', 'epoch', etc.
            is_best(bool): Si True, copie les poids dans 'best_model.pth'. Défaut False
        """
        last_path = self.ckpt_dir/"last_state.ckpt"
        torch.save(state,last_path)
        
        if is_best:
            best_path = self.ckpt_dir /"best_model.pth"
            # On ne sauvegarde que les poids pour le best pour gagner de la place
            torch.save(state.get('state_dict', state), best_path)
    
    
    def save_weak_labels(self, paths: list, labels: list):
        """
        Sauvegarde les labels faibles générées pour les données inconnues
        
        Args:
            paths(list): Liste des chemins des images de confiance
            labels(list): Liste des labels faibles associés aux paths
        """
        
        df_pseudo = pd.DataFrame({'path': paths, 'label': labels})
        df_pseudo.to_parquet(self.labels_path, index=False)
    
    
    def load_config(self, bypass:bool=False):
        """
        Charge le fichier de configuration pour avec les mêmes paramètres Modele/DataLoader
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except ConstructorError as e:
            # En présence d'objet pytorch complexe (ex: toch.device), safe_load ne fonctionnera pas
            if bypass:
                print("Chargement d'objets pytorch complexe, risque sécuritaire")
                with open(self.config_path, 'r') as f:
                    return yaml.load(f, Loader=yaml.Loader)
            else:
                print("Présence d'objets complexes.")
                raise e
    
    def load_model(self, model:nn.Module, model_name:str):
        """
        Charge les poids retenus pour un état du modèle (.pth).
        On ne charge que les poids qu'on connecte au modele (inférence/prod)!
        
        Args:
            model(nn.Module): l'instance modele
            model_name(str): l'état du modèle qu'on veut chargé (.pth) (les poids a associer)
        """
        model_path = self.ckpt_dir / model_name
        
        if not model_path.exists():
            print(f"Aucun fichier trouvé à {model_path}")
            return
        
        
        # On lit le fichier sur le disque (juste un dico de données contenant 
        # {nom_couche:tenseur de poids}) != connections aux neuronnes du modele
        # map_location permet de charger sur CPU même si sauvé sur GPU
        weights_dict = torch.load(model_path, map_location=torch.device('cpu')) 
        
        # Connecte les poids (dico) au model
        model.load_state_dict(weights_dict)
        model.eval() # Toujours passer en mode évaluation après chargement
    
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Charge un checkpoint (l'état complet d'un entrainement: modele/optimiseur/epoch).
        Parfait pour la reprise d'un entrainement.
        
        Args:
            model(nn.Module): l'instance modele
            optimizer(torch.optim.Optimizer): l'instance de l'optimiseur
        """
        ckpt_path = self.ckpt_dir / "last_state.ckpt"
        
        if not ckpt_path.exists():
            print(f"Pas de checkpoint à {ckpt_path}")
            return 0
        
        checkpoint = torch.load(ckpt_path)
        
        # Injection des poids au modele
        model.load_state_dict(checkpoint['state_dict'])
        # Chargement de l'optimiseur
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Chargement de l'epoch
        epoch = checkpoint.get('epoch', 0)
        print(f"Reprise de l'entraînement à l'epoch {epoch}")
        return epoch +1
    
