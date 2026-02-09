# imports
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
# Algorithmes de réduction de dimension. Sert à visualiser données en 2D/3D.
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# Clustering et mesure de performance non-supervisée.
# from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score,silhouette_score
from typing import Dict, List, Optional, Literal, Any, Union
import numpy.typing as npt

from notebooks.utils.clustering import cluster_model
from notebooks.utils.reducing_dim import reductor

class ClusterManager:
    """
    Gère la chaîne de traitement : Scaling -> Réduction -> Clustering -> Viz.

    Gère le pipeline post-extraction : Standardisation, Réduction de dimension,
    Clustering et Visualisation

    Cette classe permet de comparer la structure naturelle des données (clusters)
    avec les diagnostics réels (labels) pour valider la qualité des embeddings.
    
    Attributes:
        df (pd.DataFrame): Le DataFrame contenant les données et les résultats.
        feature_cols (List[str]): Liste des noms de colonnes commençant par 
            feature_prefix(str)="feature_".
        X (npt.NDArray[Any]): Matrice brute des caractéristiques.
        labels_true (npt.NDArray[Any]): Vecteur des étiquettes réelles (Ground Truth).
        X_scaled (npt.NDArray[Any]): Données normalisées (moyenne 0, variance 1).
        reductions (Dict[str, npt.NDArray[Any]]): Coordonnées issues du reducteur de dimension.
    """
    def __init__(self, features_df:pd.DataFrame, feature_prefix:str="feature_"):
        """
        Args:
            df (pd.DataFrame): Le DataFrame contenant les données et les résultats.
            feature_cols (List[str]): Liste des noms de colonnes commençant par 
                feature_prefix(str)="feature_".
            X (npt.NDArray[Any]): Matrice brute des caractéristiques.
                labels_true (npt.NDArray[Any]): Vecteur des étiquettes réelles (Ground Truth).
            X_scaled (npt.NDArray[Any]): Données normalisées (moyenne 0, variance 1).
                reductions (Dict[str, npt.NDArray[Any]]): Coordonnées issues du reducteur de dimension.
        """
        # CREEE UNE COPIE DE LA DF, A VOIR SI INDISPENSABLE
        self.df: pd.DataFrame = features_df.copy()
        # Séparation features / métadonnées
        self.feature_cols: List[str] = \
            [col for col in self.df.columns if col.startswith(feature_prefix)]
        # Tableau numpy pour les calculs sklearn
        self.X:npt.NDArray[Any] = self.df[self.feature_cols].to_numpy()
        self.labels_true:npt.NDArray[Any] = self.df['label'].to_numpy()
        
        # Scaling immédiat (toujours requis pour PCA/KMeans)
        self.X_scaled:npt.NDArray[Any] = StandardScaler().fit_transform(self.X)
        
        # Stockage des résultats de réduction (coordonnées 2D (PCA, t-SNE))
        self.reductions:Dict[str,npt.NDArray[Any]] = {} 


    def reduce_dimensions(
        self,
        # pca_params: Optional[Dict[str, Any]] = None, 
        # tsne_params: Optional[Dict[str, Any]] = None
        reductor_name:Literal["pca","tsne","isomap"] = "pca",#,"umap"]="pca",
        reductor_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, npt.NDArray[Any]]:
        """
        Applique PCA (linéaire) et t-SNE (non-linéaire) pour projeter les N features 
        vers un espace 2D visualisable. DEPRECATED
        
        Applique un réducteur de dimension
        
        Args:
            pca_params(Optional[Dict[str, Any]]): 
                dico des hyperparam pour PCA (defaut: None) DEPRECATED
            tsne_params(Optional[Dict[str, Any]]):
                dico des hyperparam pour t-SNE (défaut: None) DEPRECATED
            reductor_params(Optional[Dict[str, Any]]):
                dico des hyperparams du reducteur (défaut: None)
        """
        
        # # PCA : Capture la variance globale de manière rapide et linéaire.
        # # Utilise un algo SVD déterministe (pas besoin de random_state/seed)
        # pca = PCA(**pca_params) if pca_params else PCA()
        # self.reductions['PCA'] = pca.fit_transform(self.X_scaled)
        
        # # t-SNE : Préserve les voisinages locaux. Très puissant pour voir 
        # # si des groupes d'images sont "proches". 
        # # Est probabiliste et itératif (besoin de random_state)
        # # IMPORTANT: NE JAMAIS CLUSTEURISER SUR t-SNE qui déforme l'espace. ALGO de VISU!!
        # tsne = TSNE(**tsne_params) if tsne_params else TSNE()
        # self.reductions['t-SNE'] = tsne.fit_transform(self.X_scaled)
        
        reductor_dim = (
            reductor(reductor_name,**reductor_params) if reductor_params 
            else reductor(reductor_name)
        )
        
        self.reductions[reductor_name] = reductor_dim.fit_transform(self.X_scaled)
        
        return self.reductions


    def apply_clustering(
        self,
        reductor_name:str="pca",
        cluster_name:Literal["kmeans","dbscan","meanshift","gmm"]="kmeans",
        # kmeans_params: Optional[Dict[str, Any]] = None, 
        # dbscan_params: Optional[Dict[str, Any]] = None
        cluster_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Applique deux algorithmes de clustering (KMEANS et DBSCAN) pour découvrir des groupes.
        
        Args:
            kmeans_params(Optional[Dict[str, Any]]): 
                dico des hyperparam pour kmeans (defaut: None) DEPRECATED
            dbscan_params(Optional[Dict[str, Any]]): 
                dico des hyperparam pour dbscan (défaut: None) DEPRECATED
            cluster_params(Optional[Dict[str, Any]]): 
                dico des hyperparam (défaut: None)
            cluster_name(Literal["kmeans","dbscan","meanshift","gmm"]):
                Nom de l'algo de clustering a utiliser. par défaut: "kmeans
            reductor_name(str): Nom du reducteur de dimension. Par défaut pca
        """
        # # KMeans : Cherche des groupes sphériques. Sensible aux outliers.
        # kmeans = KMeans(**kmeans_params) if kmeans_params else KMeans()
        # self.df['cluster_kmeans'] = kmeans.fit_predict(self.reductions['PCA'])
        
        # # DBSCAN : Cherche des groupes par densité. Capable de détecter des formes complexes 
        # # et d'isoler des points aberrants (notés -1).
        # dbscan = DBSCAN(**dbscan_params) if dbscan_params else DBSCAN() 
        # self.df['cluster_dbscan'] = dbscan.fit_predict(self.reductions['PCA'])
        
        cluster = (
            cluster_model(cluster_name,**cluster_params) if cluster_params 
            else cluster_model(cluster_name)
        )
        
        self.df[f"cluster_{reductor_name}_{cluster_name}"]= \
            cluster.fit_predict(self.reductions[reductor_name])
        
        return self.df


    def evaluate_ari(
        self, 
        method:str = 'cluster_pca_kmeans'
    )->float:
        """
        Calcule l'Adjusted Rand Index (ARI) pour mesurer la concordance 
        entre les clusters trouvés et les vrais diagnostics.

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
        
        L'ARI varie de -1 à 1 (1 = correspondance parfaite, 0 = hasard).
        
        Args:
            method(str): methode de clustering a comparer "cluster_nomReducteur_nomCluster"
                (défaut: cluster_pca_kmeans)
        
        Returns:
            ari(float): score ARI
        """
        # On ne compare que ce qui est connu (ignore les 1406 non-labellisées)
        mask = self.df['label'] != -1
        subset = self.df[mask]
        
        if len(subset) == 0:
            return 0.0
            
        ari = adjusted_rand_score(subset['label'], subset[method])
        return ari


    def evaluate_silhouette(
        self,
        df_feature:Any,
        method:str = 'cluster_pca_kmeans'
    )->float:
        """
        Calcul le coef de silhouette. Ce coef prend en compte la densité des clusters ET
        l'ecart entre les clusters. Il se calcul a partir de la distance intragroupe (a) et la
        distance moyenne entre les groupes (b) tel que:
        
        coeff_silhouette = \frac{b-a}{max(a,b)}
        
        Le coeff est compris entre -1 et 1, 1 traduisant un excellent clustering.
        
        Args:
            method(str): methode de clustering a comparer "cluster_nomReducteur_nomCluster"
                (défaut: cluster_pca_kmeans)
        
        Returns:
            silhouette(float): score silhouette
        """
        labels = self.df[method]
        # La silhouette nécessite au moins 2 clusters et moins que le nombre total de points
        num_labels = len(set(labels))
        if num_labels < 2:
            return -1.0 # Score par défaut si un seul cluster (DBSCAN peut faire ça)
            
        return float(silhouette_score(df_feature, labels))


    # def plot_results(self):
    #     """Affiche les graphiques PCA et t-SNE colorés par Clusters."""
    #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
    #     # PCA - Vrais Labels
    #     sns.scatterplot(x=self.reductions['PCA'][:,0], y=self.reductions['PCA'][:,1], 
    #                     hue=self.df['label'], ax=axes[0,0], palette='viridis', style=self.df['label'])
    #     axes[0,0].set_title("PCA - Vrais Labels (Ground Truth)")
        
    #     # PCA - KMeans Clusters
    #     sns.scatterplot(x=self.reductions['PCA'][:,0], y=self.reductions['PCA'][:,1], 
    #                     hue=self.df['cluster_kmeans'], ax=axes[0,1], palette='tab10')
    #     axes[0,1].set_title("PCA - KMeans Clusters")

    #     # t-SNE - Vrais Labels
    #     sns.scatterplot(x=self.reductions['t-SNE'][:,0], y=self.reductions['t-SNE'][:,1], 
    #                     hue=self.df['label'], ax=axes[1,0], palette='viridis', style=self.df['label'])
    #     axes[1,0].set_title("t-SNE - Vrais Labels")
        
    #     # t-SNE - DBSCAN
    #     sns.scatterplot(x=self.reductions['t-SNE'][:,0], y=self.reductions['t-SNE'][:,1], 
    #                     hue=self.df['cluster_dbscan'], ax=axes[1,1], palette='tab10')
    #     axes[1,1].set_title("t-SNE - DBSCAN")
        
    #     plt.tight_layout()
    #     plt.show()