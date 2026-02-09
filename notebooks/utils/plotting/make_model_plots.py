import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
from matplotlib.figure import Figure
# import numpy as np
from typing import List, Literal, Any, Optional,Tuple, Dict
import numpy.typing as npt
from pathlib import Path
from itertools import product

from notebooks.utils.plotting.config_figures import setup_subplots, save_figure


from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve,auc

# ===========================================================================


def plot_hyperparam_effect(
    cv_results: pd.DataFrame, 
    param_prefix: str = 'param_', 
    metric_prefix: str = 'mean_test_',
    model_type: Literal['regression', 'classification'] = "regression",
    title_save: str="hyperparams",
    save_path: Path|None = None
) -> Figure:
    """
    Visualise l'impact des hyperparamètres sur les métriques de cross-validation.
    
    ENTREES:
    cv_results: DataFrame issu d'une validation croisée (ex GridSearchCV.cv_results_)
    param_prefix: Filtre pour les colonnes paramètres (ex: 'param_')
    metric_prefix: Filtre pour les colonnes métriques (ex: 'mean_test_')
    model_type: 'regression' (inverse les scores négatifs) ou 'classification'
    """
    
    # Identification dynamique des colonnes
    params = [
        col for col in cv_results.columns 
        if col.startswith(param_prefix)
    ]
    metrics = [
        col for col in cv_results.columns 
        if col.startswith(metric_prefix) 
        and 'split' not in col
    ] # On exclut les splits individuels

    # Preparation figure et axes
    n_rows, n_cols = len(metrics), len(params)
    fig, axes = setup_subplots(n_rows*n_cols,n_cols)

    # Ajout de la liste de combinaison métrique/paramètre
    tasks = [(metric, param) for metric in metrics for param in params]
    for ax, (metric, param) in zip(axes,tasks):
        param_name = param.replace(param_prefix, '')
        
        # Gestion spécifique pour l'affichage (MSE négative en Sklearn)
        y_values = cv_results[metric]
        if model_type == "regression" and (y_values < 0).any():
            y_values = -y_values
            metric_label = f"Abs({metric})"
        else:
            metric_label = metric

        # Tracé avec intervalle de confiance
        sns.lineplot(
            data=cv_results, x=param, y=y_values,
            marker='o', errorbar='sd', ax=ax
        )
        
        ax.set_title(f"{metric_label} vs {param_name}")
        ax.set_xlabel(param_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_figure(title_save,save_path)
    return fig


# ===========================================================================


def get_feature_importance(
    model: Any, 
    feature_names: List[str]|None = None,
    preprocessor: Any = None,
    X_val: pd.DataFrame|None = None, 
    y_val: pd.Series|None = None,
    method: Literal['auto','native','permutation'] = "auto"
) -> pd.Series:
    """
    Extrait les importances des features de manière générique.
    Eviter pour les réseaux de neuronnes comme MLP ou sinon réduire significativement
    le nombre d'observations (1000-2000).
    
    ENTREES:
        model: ML entraîné
        features_names:
        preprocessor: preprocessing
        X_val: dataframe des features
        y_val: series de la cible
        method: 'auto', 'native' (tree), 'permutation'
    """
    importances = None
    
    # Extraction via Permutation (Modèle agnostique)
    if method == "permutation" or (method == "auto" and not hasattr(model, "feature_importances_")):
        if X_val is None or y_val is None:
            raise ValueError("X_val et y_val sont requis pour la permutation importance.")
        
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=10,
            random_state=42,
            # n_jobs=-1
        )
        # On force Pylance à comprendre que result a l'attribut importances_mean
        # 'result' est un objet de type Bunch qui se comporte comme un dict mais avec des attributs
        importances_val = getattr(result, "importances_mean", None)
        
        if importances_val is None:
            raise AttributeError(
                "L'objet retourné par permutation_importance ne contient pas 'importances_mean'."
            )
        
        importances = pd.Series(importances_val, index=X_val.columns)

    # Extraction Native (Arbres: RF, XGBoost, CatBoost)
    else:
        # Gestion pipeline Sklearn : on cherche l'étape finale
        estimator = model.named_steps['modele'] if hasattr(model, 'named_steps') else model
        
        if hasattr(estimator, "feature_importances_"):
            raw_importances = estimator.feature_importances_
            
            # Tentative de récupération des noms de features
            final_names = feature_names
            
            # Si un preprocesseur est fourni et qu'on n'a pas les noms
            if final_names is None and preprocessor:
                try:
                    final_names = preprocessor.get_feature_names_out()
                except AttributeError:
                    # Fallback si get_feature_names_out n'existe pas 
                    # (ex: vieux sklearn ou CustomTransformer)
                    final_names = [f"feat_{i}" for i in range(len(raw_importances))]
            
            # Si toujours pas de noms, on met des indices
            if final_names is None or len(final_names) != len(raw_importances):
                final_names = [f"feat_{i}" for i in range(len(raw_importances))]
                
                importances = pd.Series(raw_importances, index=final_names)
            
        elif hasattr(estimator, "get_feature_importance"): # CatBoost spécifique
            raw_importances = estimator.get_feature_importance()
            importances = pd.Series(
                raw_importances, index=feature_names 
                if feature_names 
                else range(len(raw_importances))
            )

    if importances is None:
        raise ValueError("Impossible d'extraire l'importance des features pour ce modèle.")
        
    return importances.sort_values(ascending=False)


# ===========================================================================


def plot_feature_importance(
    importances: pd.Series, 
    top_n: int = 20, 
    title_save: str = "Feature Importance",
    save_path: Path|None = None
) -> Figure:
    """
    Affiche l'importance des features.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # On prend le Top N
    data_to_plot = importances.head(top_n)
    
    sns.barplot(
        x=data_to_plot.values, 
        y=data_to_plot.index, 
        ax=ax, 
        palette="viridis"
    )
    
    ax.set_title(title_save)
    ax.set_xlabel("Importance Relative")
    ax.grid(True, axis='x', alpha=0.5)
    
    if save_path:
        save_figure(title_save,save_path)
        
    return fig


# ===========================================================================


def pr_curve(
    y_true:np.ndarray|pd.Series, 
    y_proba:np.ndarray|pd.Series, 
)->Tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    """
    Calcul precision, recall et threshold ainsi que l'auc.
    
    Args:
        y_true: target réelles.
        y_proba: Probabilités de classe positive.
    
    """
    # thresholds renvoyé par sklearn a une longueur n_thresholds
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    
    # On aligne les vecteurs pour prec et rec 
    # (sklearn ajoute 1 et 0 à la fin de prec/rec)
    return prec[:-1], rec[:-1], thresh, float(auc(rec, prec))


# ===========================================================================


def plot_clustering(
    df:pd.DataFrame, 
    reductions:Dict[str,npt.NDArray[Any]],
    df_labels:str="cluster_"
    )->Figure:
    """
    Génère la grille de comparaison en utilisant setup_subplots et itertools.
    """
    # Préparation des éléments à croiser
    reduction_names = list(reductions.keys()) # ['PCA', 't-SNE']
    label_cols = ['label'] + [col for col in df.columns if col.startswith(df_labels)]
    
    # Toutes les combinaisons (ex: ('PCA', 'label'), ('PCA', 'cluster_kmeans'), ...)
    combinations = list(product(reduction_names, label_cols))
    num_plots = len(combinations)
    
    # Config ed la figure
    fig, axes_flat = setup_subplots(num_plots=num_plots, cols=len(label_cols))
    
    
    for ax, (reduction_name, label_name) in zip(axes_flat, combinations):
        # On ne prend que les 2 premières composantes (pour le PCA 50D)
        coords = reductions[reduction_name][:, :2]
        
        # On s'assure que les données ne contiennent plus de Tenseurs
        plot_hue = df[label_name].apply(lambda x: x.item() if hasattr(x, 'item') else x)
        plot_style = df['label'].apply(lambda x: x.item() if hasattr(x, 'item') else x)
        
        sns.scatterplot(
            x=coords[:, 0], 
            y=coords[:, 1], 
            hue=plot_hue,#df[label_name],
            style=plot_style if label_name != 'label' else None,#df['label'] if label_name != 'label' else None,
            palette='viridis' if label_name == 'label' else 'tab10',
            ax=ax,
            alpha=0.7
        )
        
        ax.set_title(f"{reduction_name} - {label_name.replace('_', ' ').title()}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    return fig


# ===========================================================================


def plot_confidence(
    probas:npt.NDArray[Any],
    threshold_range:Optional[npt.NDArray[Any]]=None,
    save_path: Optional[Path]=Path.cwd(),
    title_save:str="plot_probas",
    stats:bool=False,
)->Figure:
    """
    Plot un histogramme de la distibution de probabilité de classe par le modèle.\n
    
    3 traits en pointillé servent de référence:
        - le trait ROUGE pour le seuil standard (0.5)
        - le trait VERT pour le seuil supérieur (initial) de l'entrainement
        - le trait ORANGE pour le seuil inférieur (final) de l'entrainement
    
    On a la proba en abscisse et le nombre d'image en ordonné.
    
    
    :param probas: Le tableau des probabilités
    :type probas: npt.NDArray[Any]
    :param threshold_range: l'intervalle du threshold utilisé (decay). Défaut None
    :type threshold_range: Optional[npt.NDArray[Any]]
    :param save_path: Le chemin de sauvegarde de la figure. Par défaut dossier courant
    :type save_path: Optional[Path]
    :param title_save: le nom de la figure. par défaut plot_probas
    :type title_save: str
    :param stats: L'affichage ou non d'un résumé statistique. Par défaut False
    :type stats: bool
    :return: la figure
    :rtype: Figure
    """
    
    # Config ed la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Visualisation
    ax.hist(probas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Frontière de décision (0.5)')
    
    # Affichage des seuils
    if threshold_range:
        ax.axvline(
            x=max(threshold_range), 
            color='green', 
            linestyle=':', 
            label=f'Seuil initial ({max(threshold_range)})'
        )
        ax.axvline(
            x=min(threshold_range), 
            color='orange', 
            linestyle=':', 
            label=f'Seuil final ({min(threshold_range)})'
        )
    
    ax.set_title("Distribution des scores de confiance sur le jeu inconnu")
    ax.set_xlabel("Probabilité prédite (0 = Sain, 1 = Cancer)")
    ax.set_ylabel("Nombre d'images")
    ax.legend()
    
    # Statistiques
    if stats:
        print(f"--- Statistiques des scores ---")
        print(f"Médiane : {np.median(probas):.4f}")
        print(f"Moyenne : {np.mean(probas):.4f}")
        print(f"Images entre 0.4 et 0.6 (incertaines) : {np.sum((probas > 0.4) & (probas < 0.6))}")
        print(f"Images > 0.95 (très sûres) : {np.sum(probas > 0.95)}")
        
    plt.tight_layout()
    if save_path:
        save_figure(title_save,save_path)
    return fig