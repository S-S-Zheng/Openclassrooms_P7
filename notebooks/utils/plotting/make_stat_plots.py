import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# import numpy as np
from typing import List, Optional, Literal
from pathlib import Path

from notebooks.utils.plotting.config_figures import setup_subplots, save_figure
from notebooks.utils.features_type_list import features_type


# ===================== Fonctions internes ========================
def _plot_scatter(df: pd.DataFrame, x: str, y: str, ax: Axes, scale: bool):
    """Fonction interne pour tracer un scatter plot."""
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    if scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(f"{y} vs {x}")

def _plot_box(df: pd.DataFrame, x: Optional[str], y: str, ax: Axes, scale: bool):
    """Fonction interne pour tracer un boxplot."""
    sns.boxplot(
        data=df, x=x, y=y, ax=ax,
        medianprops={"color": "r", "linewidth": 2}
    )
    if scale:
        ax.set_yscale("log")
    ax.set_title(f"Distribution de {y}" + (f" par {x}" if x else ""))

def _plot_hist(df: pd.DataFrame, col: str, ax: Axes):
    """Fonction interne pour tracer un histogramme."""
    # Détection automatique de l'orientation (numérique vs catégorielle)
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    
    sns.histplot(
        data=df, 
        x=col if is_numeric else None,
        y=None if is_numeric else col,
        ax=ax
    )
    ax.set_title(f"Distribution de {col}")


# ============================================================================


def plot_scat_box_hist(
    df: pd.DataFrame,
    plot_type: Literal['scatter', 'box', 'hist'],
    features: List[str],
    x_axis: str|None = None,
    scale: bool = False,
    title_save:str = "scatter_box_hist",
    save_path: Path|None = None
) -> None|Figure:
    """
    Génère une grille de graphiques (Scatter, Box ou Hist) pour une liste de features.
    
    ENTREES:
    df: DataFrame
    plot_type: 'scatter', 'box', 'hist'
    features: Liste des variables à visualiser (axe Y pour scatter/box)
    x_axis: Variable axe X (obligatoire pour scatter, optionnel pour box)
    scale: Si True, applique une échelle log
    save_path: Chemin complet pour sauvegarder l'image (sans extension)
    
    SORTIES:
    fig: La figure Matplotlib générée
    """
    if not features:
        print("Aucune feature fournie pour le plot.")
        return None

    # Création de la grille
    fig, axes = setup_subplots(len(features))
    
    for ax, feature in zip(axes, features):
        if plot_type == "scatter":
            if not x_axis:
                raise ValueError("L'argument 'x_axis' est requis pour les scatter plots.")
            _plot_scatter(df, x_axis, feature, ax, scale)
            
        elif plot_type == "box":
            _plot_box(df, x_axis, feature, ax, scale)
            
        elif plot_type == "hist":
            _plot_hist(df, feature, ax)

    fig.tight_layout()
    
    if save_path:
        save_figure(title_save, save_path)
    return fig


# ===========================================================================


def plot_pairplot(
    df: pd.DataFrame,
    hue: str|None = None,
    title_save:str="pairplot",
    save_path: Path|None = None,
    # Options pour le pairplot
    kind: Literal["scatter","reg"] = "scatter",
    diag_kind: Literal['auto', 'hist', 'kde'] | None = "auto"
) -> sns.axisgrid.PairGrid:
    """
    Wrapper spécifique pour le pairplot de Seaborn (qui gère sa propre figure).
    
    ENTREES:
    df: DataFrame
    hue:  Feature de regroupement: une couleur = une catégorie. Doit être dans df!
    title_save: Nom du fichier pour la sauvegarde
    save_path: Chemin complet pour sauvegarder l'image (sans extension)
    kind: Définie le type de graphique pour chaque paire de variable (hors diagonale)
    ==> scatter pour le nuage de points et reg pour scatter+regression.
    diag_kind: définie la distribution univariée sur la diagonale
    ==> hist pour histogramme, kde pour densité
    
    SORTIES:
    fig: La figure Matplotlib générée
    """
    # Sélection des numériques uniquement pour éviter les plantages
    num_list, _ = features_type(df)
    df_num = df[num_list]
    
    # Si hue est fourni, on s'assure qu'il est dans le dataframe
    if hue and hue in df.columns:
        df_num = df_num.copy()
        df_num[hue] = df[hue]
    
    fig = sns.pairplot(
        df_num,
        hue=hue,
        kind=kind,
        diag_kind=diag_kind if diag_kind != "auto" else "kde"
    )

    
    if save_path:
        save_figure(title_save, save_path)
    return fig