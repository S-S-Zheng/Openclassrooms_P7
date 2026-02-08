

# imports
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture


def cluster_model(
    cluster_name:str, 
    **kwargs
):
    """
    Instancie un algorithme de clustering pour l'imagerie parmi les choix suivants:
        KMeans: Methode classique; divise l'image en K clusters en minim la variance intra-classe
            Très rapide mais nécéssite de choisir K à l'avance.
            (Besoin de K, Très sensible au bruit, cluster sphérique)
        DBSCAN: Se base sur la densité locale pour regrouper les points. Les zones denses sont
            identifiés comme cluster tandis que les faible dénsifiées sont associés au bruit.
            Ignore efficacement le bruit mais asocie les cas particuliers ou forte variance au bruit.
            (Pas besoin de K, peu sensible au bruit, cluster arbitraire)
        Mean Shift: Contrairement au KMeans, le Mean Shift est associé a une méthode non paramétrique
            qui ne nécéssite pas de connaitre à l'avance le nombre de cluster. Il prend en compte
            la distribution de s pixels comme une fonction de densité de prob et déplace itérativ
            les points vers les zones a plus forte densité (mode). Est performant pour déter le nb
            de cluster et gère bien les formes arbitraires mais est très gourmand en puissance.
            (Pas besoin de K, très peu sensible au bruit, cluster arbitraire)
        GMM: Approche statistique. Suppose que les données sont issus d'un mélange de plusieurs
            distributions gaussiennes ou chaque pixel a une proba d'appartenir a un cluster.
            très flexible sur la forme des clusters mais a bien initialiser sinon risque 
            de converger vers un max local.
            (Besoin de K, sensible au bruit, cluster ellipsoidale)
    
    Args:
        cluster_name(str): nom du cluster a instancier
    """
    models ={
        "kmeans":KMeans,
        "dbscan":DBSCAN,
        "meanshift":MeanShift,
        "gmm":GaussianMixture,
    }
    
    cluster_name=cluster_name.lower()
    
    if cluster_name not in models:
        raise ValueError(f"Algo pas inclu dans la liste: {list(models.keys())}")
    
    return models[cluster_name](**kwargs)