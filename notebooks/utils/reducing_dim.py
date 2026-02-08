

# imports
# Algorithmes de réduction de dimension. Sert à visualiser données en 2D/3D.
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,Isomap
# import umap

def reductor(
    reductor_name:str, 
    **kwargs
):
    """
    Instancie un algorithme de reduction de dimensions pour l'imagerie parmi les choix suivants:
        PCA: Projette les données sur des axes qui maximisent la variance. transformation linéaire.
            Extrêmement rapide, déterministe et permet inverse transform mais très mauvais si la
            donnée est courbe. C'est le standard d'utilisation avant le clustering.
            (lineaire, idéale pour pré-processing/compression, extremement rapide)
        t-SNE: Convertit les dist euclidiennes en proba de voisinnage en déformant l'espace. 
            Excellent pour la visualisation et la séparation de clusters complexes mais très lent
            et les distances entre clusters ne veulent plus rien dire (donc inexploitable clustering)
            Eclate les géométries et n'est donc pas adapté au clustering.
            (Non-linéaire, idéale pour visu, très lent)
        Isomap: Cherche a preserver les dist suivant la forme du nuage de point plutot que la ligne
            droite. Excellent pour déplier des structure non lineaires mais très sensible au bruit
            si les points sont trop espacés. Si les données sont sur une trajectoire continue,
            le clustering post isomap peut etre recommandé (ex evolution d'une tumeur).
            (géométrique, ideal pour la comprehension géométriue des données, moyennement rapide)
        UMAP: Similaire au t-SNE mais beaucoup plus rapide et préserve mieux la structure globale
            des données. Excellent pour le clustering SI les données on une forme complexe 
            (exemple detection de tumeur au cerveau!, IMPLEMENTATION NATIVE SEMI-SUPERVISED).
            (Topologique, idéal pour la visu et la perf, très rapide)
    
    Args:
        reductor_name(str): nom du reducteur de dimension a instancier
    """
    models ={
        "pca":PCA,
        "tsne":TSNE,
        "isomap":Isomap,
        # "umap":umap.UMAP, # A revoir, problème de dépendance...
    }
    
    reductor_name=reductor_name.lower()
    
    if reductor_name not in models:
        raise ValueError(f"Algo pas inclu dans la liste: {list(models.keys())}")
    
    return models[reductor_name](**kwargs)