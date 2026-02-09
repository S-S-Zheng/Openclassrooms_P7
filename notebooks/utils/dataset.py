"""
Charger les données, gérer les chemins et appliquer les transformations.
"""

# imports
from pathlib import Path
# La bibliothèque standard pour ouvrir les fichiers image (.jpg, .png).
from PIL import Image
# Créer tenseurs rempli de 0
# from torch import zeros,Tensor
# Pour créer une structure qui lit images et les envoie par paquets (batches) au GPU.
from torch.utils.data import Dataset
# Pour redimensionner, normaliser ou augmenter images (rotation, etc.).
from torchvision import transforms
from typing import List, Any, Union, Optional,Tuple, Sequence,Literal


# ============================================================================


class BaseTransform:
    """
    Base class de préparatoin et transformation des images
    
    Attributes:
        mean (List[float]): Moyenne par canal pour la normalisation.
        std (List[float]): Écart-type par canal pour la normalisation.
        size (int): Dimension (H, W) cible pour le redimensionnement.
    """
    # ============ Attributs/données de la classe (ce qu'elle est) =========
    def __init__(
        self, 
        mean:List[float]=[0.5,0.5,0.5],
        std:List[float]=[0.5,0.5,0.5],
        size:int=224
    ):
        self.mean = mean
        self.std = std
        self.size = size

    # ============ Méthodes/actions de la classe (ce qu'elle fait) ========
    def preproc(
        self, 
        train:bool=False,
        # horiz_flip:float = 0.5,
        # rotation:int = 10,
        # brightness:float | tuple[float, float] = 0.2,
        # contrast:float | tuple[float, float] = 0.2,
        # saturation:float | tuple[float, float] = 0.2,
        strong_augment:bool=False,
        **kwargs
    )->transforms.Compose:
        """
        Pipeline de transformation Pytorch\n
        Prépare les images: redimensionne et normalise.
        Si c'est le jeu d'entrainement: augmentation en plus auquel on applique ou non une forte augm
        
        Paramètres kwargs possible pour l'augmentation du train:
        
            - h_flip = kwargs.get('horiz_flip', 0.5)
            - rot = kwargs.get('rotation', 10)
            - bright = kwargs.get('bright', 0.2)
            - contrast = kwargs.get('contrast', 0.2)
            - saturation = kwargs.get('saturation',0.2)
            - crop = kwargs.get('crop',(0.08,1))
            - erasing = kwargs.get('erasing',0.5)
        
        Args:
            train(bool) :Inclure les augmentation si c'est le train ou pas. défaut: False
            # horiz_flip(float): inclinaison horizontale. défaut = 0.5,
            # rotation(int): Angle de rotation défaut = 10,
            # brightness(float | tuple[float, float]): luminosité, défaut = 0.2,
            # contrast(float | tuple[float, float]): contraste, défaut = 0.2,
            # saturation(float | tuple[float, float]): saturation, défaut = 0.2,
            strong_augment(bool): Si True, forte augmentation appliqué au jeu train. défaut False
        
        Returns:
            transforms.Compose: Un objet callable appliquant les transformations.

        """
        tf_list:List[Any] = [transforms.Resize((self.size, self.size))] # Redimensionne
        
        if train:
            # Data Augmentation seulement pour l'entraînement
            # tf_list.extend([
            #     transforms.RandomHorizontalFlip(p=horiz_flip),
            #     transforms.RandomRotation(degrees=rotation),
            #     transforms.ColorJitter(
            #         brightness=brightness, 
            #         contrast=contrast,
            #         saturation=saturation
            #     ),
            # ])
            # Paramètres par défaut avec kwargs.get(clé, défaut)
            h_flip = kwargs.get('horiz_flip', 0.5)
            rot = kwargs.get('rotation', 10)
            
            tf_list.append(transforms.RandomHorizontalFlip(p=h_flip))
            tf_list.append(transforms.RandomRotation(degrees=rot))
            
            if strong_augment:
                # On ajoute des augmentations importantes
                bright = kwargs.get('bright', 0.2)
                contrast = kwargs.get('contrast', 0.2)
                saturation = kwargs.get('saturation',0.2)
                crop = kwargs.get('crop',(0.08,1))
                
                
                tf_list.extend([
                    transforms.ColorJitter(
                        brightness=bright,
                        contrast=contrast,
                        saturation=saturation
                    ),
                    transforms.RandomResizedCrop(self.size, scale=crop),
                ])
            
        # Rearque: cette partie est détachée car si train, il faut d'abord augmenter avant
        # de tenseuriser et normaliser
        tf_list.extend([
            transforms.ToTensor(), # Transforme l'image PIL en numérique(0.0 a 1.0)
        ])
        
        # ===== ON DOIT PLACER L'ERASING APRES LA TENSEURISATION (CONTRAINTE TORCHVISION) ===
        # la méthode diffère du reste qui travail sur PIL, erasing remplace les valeurs du tenseur
        # par 0 == efface l'image/ noirci
        if strong_augment:
            erasing = kwargs.get('erasing',0.5)
            tf_list.extend([
                transforms.RandomErasing(p=erasing)
            ])
        
        tf_list.extend([
            transforms.Normalize(self.mean, self.std) # Normalise les images
        ])
        return transforms.Compose(tf_list)


# ============================================================================


class ImagesToDataset(Dataset):
    """
    Subclass de la class Dataset de pytorch. Nécéssaire car DataLoader (gère le multi threading
    et le mélange de données) ne fonctionne QUE si la classe "est un" Dataset.\n
    Gère le chargement des images et des labels.

    Les méthodes Dunder (__) ci-dessous permettent aux objets de se comporter comme des structures
    natives, par exemple:\n
        __len__ permet de faire len(mon_dataset)\n
        __getitem__ permet d'acceder a l'image avec des crochets image = mon_dataset[5]\n
    Ces deux dunder sont indispensables pour utiliser DataLoader.
    
    Attributes:
        file_paths: Liste des chemins vers les fichiers images.
        labels: Liste des classes (index ou noms).
        transform: Pipeline de transformations (ex: BaseTransform.preproc())
    """
    def __init__(
        self, 
        file_paths:List[Path], 
        labels:Sequence[Union[str,int]], # Sequence est immutable != List qui l'est
        transform:Optional[transforms.Compose]=None
    ):
        self.file_paths = file_paths # Liste des chemins des images
        self.labels = labels # Liste de strings ou d'entiers
        self.transform = transform # Composition de la class BaseTransform

    def __len__(self)->int:
        """
        Nombre total d'images.
        
        Args:
            file_paths: Liste des chemins vers les fichiers images.
            labels: Liste des classes (index ou noms).
            transform: Pipeline de transformations (ex: BaseTransform.preproc())
        """
        return len(self.file_paths)

    def __getitem__(self, idx:int)->Tuple[Any, Any, str]:
        """
        Récupère l'image et le label à l'index donné.
        
        Returns:
            Tuple (tenseur_image, label, chemin_du_fichier)
        """
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Retourner le chemin en tant que chaîne (str) évite que DataLoader
        # tente de coller des objets pathlib.Path (ce qui déclenche
        # TypeError: default_collate: batch must contain tensors, ... found Path).
        return image, label, str(path) # Si fonction de perte type nn.CrossEntropyLoss label en int!