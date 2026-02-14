#imports
import pandas as pd
import joblib
import json
from pathlib import Path
from itertools import product
from typing import Any,Literal,Optional,List
import yaml
import gc

# ===========================================================================


def save_datas(
    data:Any, 
    folder_path:Path, 
    subs:str="", 
    filename:str="", 
    format:Literal["parquet","joblib","json","csv","yml","yaml"]="csv"
)-> Optional[Path]:
    """
    Eporte la dataframe vers le chemin spécifié
    
    Args:
        data(Any): donnée a sauvegarder
        folder_path(Path): chemin du dossier a destination
        subs(str): sous-dossier
        filename(str): nom du fichier
        format(Literal["parquet","joblib","json","csv","yml","yaml"]): 
            format de sauvegarde. par défaut: csv
    
    Returns:
        path(Path): chemin
    """
    path = Path(folder_path/subs)
    
    # Création du dossier si inexistant
    path.mkdir(parents=True,exist_ok=True)
    
    filename_path = path/f"{filename}.parquet"
    
    try:
        if format == "parquet":
            data.to_parquet(filename_path, index=False)
        elif format == "joblib":
            joblib.dump(data,filename_path)
        elif format == "json":
            if isinstance(data,pd.DataFrame):
                data.to_json(filename_path,orient="records",indent=4)
            else:
                with open(filename_path,"w") as f:
                    json.dump(data,f,indent=4)
        elif format == "csv":
            data.to_csv(filename_path, index=False)
        elif format == "yaml" or format == "yml":
            with open(filename_path,"w") as f:
                yaml.dump(data,f,indent=4)
        else:
            return None
    except Exception as e:
        print(f'Erreur, sauvegarde avortée: {e}')
        return None
    return path


# ===========================================================================


def combinaisonAB(a:Any,b:Any)->Any:
    """
    Génère toutes les combinaisons possible entre a et b. Très flexible et est équivalent
    a une double boucle for donc faire attention a la taille de a et b!
    
    a et b n'ont pas besoin d'avoir la même taille et peuvent etre:
    
    Listes ou Tuples : product([1, 2], ('a', 'b'))
    Sets (Ensembles) : product({1, 2}, {3, 4})
    Strings (Chaînes) : product("AB", "12") $\to$ (A,1), (A,2), (B,1), (B,2)
    Dictionnaires : Si tu passes un dictionnaire, il itère par défaut sur les clés.
    Générateurs : Il peut même consommer des objets qui ne sont pas encore en mémoire.
    
    Args:
        a(Any)
        b(Any)
    Returns:
        Any
    """
    return product(a,b)


# ===========================================================================


def get_score(
    f2:float, 
    ece:float, 
    pr_auc:float,
    mode:Optional[Literal['ssl','supervised','cluster']]="ssl"
):
    """
    scoring suivant f2,PR AUC et ECE. 

    Args:
        f2(float): f2
        ece(float): ece
        pr_auc(float): pr auc
        mode(Optional[Literal['ssl','supervised','cluster']]): le mode de scoring, 
            en ssl on est strict sur l'ECE et sinon on privilégie F2. défaut ssl.
            (Les poids restent arbitraires mais ne pas oublier que les metriques fluctue diff)

    Returns:
        _type_: score
    """
    if mode == "ssl":
        score = (f2 * 0.75 + pr_auc * 0.25) - (0.75 * ece)
    else:
        score = (f2 * 0.75 + pr_auc * 0.25) - (0.25 * ece)
    return score


# =======================================================================

def clean_var(var:List[str]|str)->None:
    """
    Applique del sur les noms des variables indiquées
    """
    
    if len(var)>1:
        for i in var:
            if i in locals():
                del locals()[i]
            if i in globals():
                del globals()[i]
    else:
        del var