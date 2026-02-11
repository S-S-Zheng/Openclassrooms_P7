# imports
import numpy as np
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
        
        self.df[f"{reductor_name}_{cluster_name}"]= \
            cluster.fit_predict(self.reductions[reductor_name])
        
        return self.df


    def evaluate_ari(
        self, 
        method:str = 'pca_kmeans'
    )->float:
        """
        Calcule l'Adjusted Rand Index (ARI) pour mesurer la concordance 
        entre les clusters trouvés et les vrais diagnostics.

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
        
        L'ARI varie de -1 à 1 (1 = correspondance parfaite, 0 = hasard).
        
        Args:
            method(str): methode de clustering a comparer "nomReducteur_nomCluster"
                (défaut: pca_kmeans)
        
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
        reductor_name: str = 'pca',
        cluster_name: str = 'kmeans'
    )->float:
        """
        Calcul le coef de silhouette. Ce coef prend en compte la densité des clusters ET
        l'ecart entre les clusters. Il se calcul a partir de la distance intragroupe (a) et la
        distance moyenne entre les groupes (b) tel que:
        
        coeff_silhouette = \frac{b-a}{max(a,b)}
        
        Le coeff est compris entre -1 et 1, 1 traduisant un excellent clustering.
        
        Args:
        #     method(str): methode de clustering a comparer "nomReducteur_nomCluster"
        #         (défaut: pca_kmeans)
            reductor_name(str): Nom de la méthode de réductions
            cluster_name(str): Nom de la méthode de clustering
        
        Returns:
            silhouette(float): score silhouette
        """
        labels = self.df[f"{reductor_name}_{cluster_name}"]
        data_coords = self.reductions.get(reductor_name)
        if data_coords is None: 
            return -1.0 # pour faire plaisir a Pylance
        
        # La silhouette nécessite au moins 2 clusters et moins que le nombre total de points
        num_labels = len(set(labels))
        if num_labels < 2:
            return -1.0 # Score par défaut si un seul cluster (DBSCAN peut faire ça)
        
        # Retrait du bruit sur DBSCAN
        mask = labels != -1
        if np.sum(mask) < 2: return -1.0
        
        return float(silhouette_score(data_coords[mask], labels[mask]))


    def cluster_pseudo_labels(
            self, 
            method: str = "pca_kmeans"
        ) -> pd.DataFrame:
            """
            Assigne un pseudo-label aux données non étiquetées (-1) en se basant sur 
            le label majoritaire du cluster auquel elles appartiennent.
            
            Logique :
            1. Pour chaque cluster, on regarde les points qui ont un vrai label (0 ou 1).
            2. On identifie le label majoritaire dans ce cluster.
            3. On assigne ce label à tous les points '-1' de ce même cluster.
            
            Args:
                method (str): La colonne de clustering à utiliser (ex: 'pca_kmeans').
                
            Returns:
                pd.DataFrame: Le DataFrame avec une nouvelle colonne 'cluster_pseudo_label'.
            """
            if method not in self.df.columns:
                raise KeyError(f"La colonne {method} n'existe pas. Lancez apply_clustering d'abord.")

            # Initialisation avec les vrais labels
            self.df['cluster_pseudo_label'] = self.df['label']
            
            # On itère sur chaque cluster identifié par l'algorithme
            unique_clusters = [c for c in self.df[method].unique() if c != -1] # On ignore le bruit
            
            for cluster_id in unique_clusters:
                # On récupère les lignes du cluster qui ont un vrai label
                mask_cluster = self.df[method] == cluster_id
                known_labels_in_cluster = self.df[mask_cluster & (self.df['label'] != -1)]['label']
                
                if not known_labels_in_cluster.empty:
                    # On trouve le label majoritaire (Mode)
                    majority_label = known_labels_in_cluster.value_counts().idxmax()
                    
                    # On assigne ce label aux inconnus de ce cluster uniquement
                    mask_unlabeled = mask_cluster & (self.df['label'] == -1)
                    self.df.loc[mask_unlabeled, 'cluster_pseudo_label'] = majority_label
                    
                    print(f"Cluster {cluster_id} mappé au Label {majority_label} "
                        f"({len(known_labels_in_cluster)} refs)")
                else:
                    print(f"Cluster {cluster_id} ignoré (aucun label de référence)")

            return self.df