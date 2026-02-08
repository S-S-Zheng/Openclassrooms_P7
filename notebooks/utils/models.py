"""
Fournir les architectures neuronales (Feature Extractor et Classifieur).
"""

# imports
import torch
import torch.nn as nn
# Contient des architectures pré-entraînées (ResNet, VGG).
from torchvision import models


# ============================================================================

# EXTRACTEUR
class FeatureExtractor(nn.Module):
    """
    Dans PyTorch, absolument tout ce qui touche aux réseaux de neurones
    (une couche, un bloc ou le modèle entier) doit hériter de nn.Module.\n
    Extraits les features pour en faire des vecteurs visuels(embeddings) qui pourront être
    lu est compris par les algo de réduction de dimension et de clustering (PCA,t-SNE,K-Means,DBSCAN).
    
    Attributes:
        weights (Literal[ResNet34_Weights.IMAGENET1K_V1] | None): Charge les poids
        base_model (ResNet): Modele de Deep Learning chargé pour le transfer learning
        backbone (nn.Sequential): L'architecture ResNet amputée de sa couche de décision finale.
    
    """
    def __init__(self, pretrained:bool=True):
        """
        Initialise l'extracteur.
        
        Args:
            pretrained (bool): Si True, charge les poids entraînés sur ImageNet. 
                Fortement recommandé pour les petits datasets.
            weights (Literal[ResNet34_Weights.IMAGENET1K_V1] | None): Charge les poids
            base_model (ResNet): Modele de Deep Learning chargé pour le transfer learning
            backbone (nn.Sequential): L'architecture ResNet amputée de sa couche de décision finale.
        """
        super().__init__()
        base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
        
        # Transfer learning: On va faire passer ResNet d'un classifieur à un extracteur de features
        # ==> On garde tout sauf la couche finale (fc la couche de classification)
        # On utilise .children())[:-1] pour récupérer les blocs 
        # (càd couche d'entrée / blocs de convolution / couche de pooling) sans la dernière couche fc
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Transforme une image en vecteur de features (embedding).\n
        Ce n'est pas une méthode Dunder MAIS le nom reste figé car nn.Module utilise un Dunder 
        __call__ qui lui cherche forward!\n
        
        IMPORTANT: LES 512 features sont pour un nb de couvhe de 18 ou 34. sinon ca peut etre 1280,
            2048 ou 4096 selon les modèles!
        
        Args:
            x (torch.Tensor): Batch d'images de forme (Batch, 3, 224, 224)->(Batch,RGB,size,size).
            
        Returns:
            torch.Tensor: Vecteurs d'embeddings de forme (Batch, 512). 512 est une valeur arbirtraire
            choisie par les dev de ResNet.\n
                Ce (512,1,1) découle des couches de convolution et de 
                Pooling. La convolution va augmenter le nombre caractéritiques issu de RGB.
                On passe ainsi de 3 à 64 puis 128, 256 et 512. Le pooling divise par 2 les dimensions
                toutes les n couches ainsi on passe de 224 a 112, 56, 28, 14, 7, on obtient alors un
                cube de (512,7,7) et parce qu'on moyenne chaque caractéristiques suivant 
                les pixels on a alors (512,1,1)
        """
        # Sortie du backbone : (Batch, 512, 1, 1)
        x = self.backbone(x)
        # flatten() aplati tout en un or on le tenseur:(images,features,taille_pixel,taille_pixel)
        # On veut créer des obsevations (ligne) pour chaque image donc (images, features)
        return torch.flatten(x,1) # Aplatit (Batch, 512, 1, 1) -> (Batch, 512)


# ============================================================================

# CLASSIFICATEUR
class BrainCancerClassifier(nn.Module):
    """
    Modèle de classification (CNN amputé ResNet auquel on attache une nouvelle couche de classif).
    
    IMPORTANT: Les couches BatchNorm et Dropout se comportent différemment 
    en train et en test/validation\n
    ==> il faut les FIXER via model.train() avant l'entrainement et model.eval avant la val/test
    
    Attributes:
        base_model (ResNet): Modele de Deep Learning chargé pour le transfer learning
        backbone (nn.Sequential): L'architecture ResNet amputée de sa couche de décision finale.
    """
    def __init__(
        self, 
        num_classes:int=1, # binaire 1 si Sigmoid, 2 si Softmax
        freeze_backbone:bool=True,
        n_features:int=512,
        dropout:float=0.5
    ):
        """
        Args:
            num_classes(int): Nombre de classes. défaut 1 (binaire)
            freeze_backbone(bool): Pour geler les couches du modèle si True. Défaut True
            n_features(int): Correspond au nombre de feature renvoyé par le CNN. Défaut 512
            dropout(float): ratio de noeud neuronale a désact aléatoiremt (evite par coeur): Défaut:0.5
        """
        super().__init__()
        base_model = models.resnet34(weights='DEFAULT')
        
        # On décapite l'extracteur (retrait de a derniere couche fc)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # au lieu de passer par no_grad(), on gele directement l'extracteur ici
        # on gèle toutes couches
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # On dégèle la dernière couche restante afin de permettre au modèle d'ajuster
            # ses poids (ré apprendre) sur la compréhension avancée tout en gardant ses bases intact
            # ==> Améliore le modèle théoriquement
            for param in self.backbone[-1].parameters():
                param.requires_grad = True
        
        # On définit la tête de classification séparément
        half_features = int(n_features/2)
        self.fc = nn.Sequential(
            nn.Flatten(), # Aplati le cube de donnée (comme torch.flatten(x,1)): OBLIGATOIRE
            nn.Linear(
                n_features, 
                half_features
            ), # 1ere couche de calcul qui combine # les features : OBLIGATOIRE
            nn.BatchNorm1d(
                half_features
            ), # Normalise sorties du Linear pour aider la convergence: OPTIONNEL
            nn.ReLU(), # Fonction d'activation qui on/off les neurones: OBLIGATOIRE
            nn.Dropout(
                dropout
            ), # off ratio de neurones aléatoiremt -> éviter le "par coeur" : OPTIONNEL
            nn.Linear(
                half_features, 
                num_classes
            ), # Comrpime les 256 neurnes à la rép fin: OBLIGATOIRE
            nn.Sigmoid() # Standardise la réponse finale entre 0 et 1: OBLIGATOIORE
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        Transforme une image en vecteur de features (embedding).\n
        Ce n'est pas une méthode Dunder MAIS le nom reste figé car nn.Module utilise un Dunder 
        __call__ qui lui cherche forward!\n
        
        IMPORTANT: LES 512 features sont pour un nb de couvhe de 18 ou 34. sinon ca peut etre 1280,
            2048 ou 4096 selon les modèles!
        
        Args:
            x (torch.Tensor): Batch d'images de forme (Batch, 3, 224, 224)->(Batch,RGB,size,size).
            
        Returns:
            classifier(torch.Tensor): CNN
        """
        # Extraction des caractéristiques (le cube 512,1,1)
        extractor = self.backbone(x)
        # Fixation de la tête fc pour classifier
        classifier = self.fc(extractor)
        return classifier