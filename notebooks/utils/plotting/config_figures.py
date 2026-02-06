
#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Literal
from pathlib import Path


# ===========================================================================


def setup_subplots(
    num_plots: int,
    cols: int = 2,
    sharex:bool | Literal['none', 'all', 'row', 'col']=False,
    sharey:bool | Literal['none', 'all', 'row', 'col']=False,
    cols_to_rows = False
) -> Tuple[Figure, np.ndarray]:
    """
    Configure une grille de subplots dynamique en fonction du nombre de variables.
    
    ENTREES:
    num_plots: Nombre total de graphiques à afficher
    cols: Nombre de colonnes désirées (défaut 2)
    sharex,sharey: partager meme axe X/Y
    cols_to_rows: permuter la représentation de la figure
    via la permutation cols et rows (seulement pour la représentation)
    
    SORTIES:
    fig: L'objet Figure
    axes: Tableau d'axes aplati (1D array)
    """
    if num_plots == 0:
        return plt.figure(), np.array([])
        
    cols = 1 if num_plots == 1 else cols
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(8*cols, 6*rows) if not cols_to_rows else (8*rows, 6*cols),
        sharex=sharex, sharey=sharey
    )
    
    # On aplatit les axes pour faciliter l'itération, même s'il n'y a qu'un plot
    axes_flat = np.array(axes).reshape(-1) # flatten
    
    # On supprime les axes vides excédentaires
    for i in range(num_plots, len(axes_flat)):
        fig.delaxes(axes_flat[i])
        
    return fig, axes_flat


# ===========================================================================


def style_heatmap_axis(ax: Axes, title: str) -> None:
    """
    Applique le style spécifique aux axes du graphique.
    Responsabilité : Mise en forme (Styling).
    """
    ax.set_title(title, fontsize=12)
    
    # Paramètres communs pour les axes X et Y
    common_params = {
        'labelsize': 10,
        'length': 6,
        'width': 2,
        'colors': 'r',
        'grid_color': 'r',
        'grid_alpha': 0.5
    }
    
    ax.tick_params(axis='x', labelbottom=True, **common_params)
    ax.tick_params(axis='y', **common_params)


# ===========================================================================


def save_figure(title: str,path:Path|None=None) -> None:
    """
    Sauvegarde la figure courante.
    Responsabilité : Entrées/Sorties (I/O).
    """
    base = path if path is not None else Path.cwd()
    fname = base/title
    
    if path is not None:
        Path.mkdir(base,exist_ok = True)
    
    plt.savefig(
        fname=fname,
        dpi=300,
        format='png',
        bbox_inches='tight'
    )
    print(f"{title} sauvegardé dans {base}")

