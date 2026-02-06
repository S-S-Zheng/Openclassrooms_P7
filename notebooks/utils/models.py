"""
Fournir les architectures neuronales (Feature Extractor et Classifieur).
"""

# imports
import torch
import torch.nn as nn
# Contient des architectures pré-entraînées (ResNet, VGG).
from torchvision import models


# ============================================================================


class FeatureExtractor(nn.Module):
    """
    Dans PyTorch, absolument tout ce qui touche aux réseaux de neurones
    (une couche, un bloc ou le modèle entier) doit hériter de nn.Module.\n
    Extraits les features pour en faire des vecteurs visuels(embeddings) qui pourront être
    lu est compris par les algo de réduction de dimension et de clustering (PCA,t-SNE,K-Means,DBSCAN).
    
    Attributes:
        weights (Literal[ResNet18_Weights.IMAGENET1K_V1] | None): Charge les poids
        base_model (ResNet): Modele de Deep Learning chargé pour le transfer learning
        backbone (nn.Sequential): L'architecture ResNet amputée de sa couche de décision finale.
    
    """
    def __init__(self, pretrained:bool=True):
        """
        Initialise l'extracteur.
        
        Args:
            pretrained (bool): Si True, charge les poids entraînés sur ImageNet. 
            Fortement recommandé pour les petits datasets.
        """
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base_model = models.resnet34(weights=weights)
        
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


class BrainTumorClassifier(nn.Module):
    """
    Modèle complet pour la classification (Fine-tuning).
    """
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT
        self.backbone = models.resnet34(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remplacement de la tête de classification
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)