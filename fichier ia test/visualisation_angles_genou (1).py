# -*- coding: utf-8 -*-

# %%
# # Visualisation de l'angle du genou droit
#
# Deux fonctions :
# 1. afficher_angles(...)         -> angle en fonction du temps (une courbe par axe)
# 2. afficher_effet_calibration(...) -> comparaison brut vs calibré, pour voir
#    visuellement l'écart introduit par la soustraction de la ROM
#
# Ces fonctions réutilisent temps, angles, angles_cal, angles_ref déjà
# calculés dans le script précédent (projet_chgmt_repere_corrige.py).

# %%
import numpy as np
import matplotlib.pyplot as plt

TITRES = ["Flexion (+) / extension (-)",
          "Abduction (+) / adduction (-)",
          "Rotation interne (+) / externe (-)"]
COULEURS = ["tab:red", "tab:green", "tab:blue"]


# %%
# ## 1. Angle en fonction du temps
#
# Affiche les 3 composantes de l'angle du genou (flexion, abduction,
# rotation) en fonction du temps, sur une même figure (3 sous-graphiques
# empilés, axe des temps partagé).

def afficher_angles(temps, angles, titre_general=None, fichier=None):
    """
    temps   : array (n_frames,) en secondes
    angles  : array (3, n_frames) en degrés, typiquement `angles` (brut)
              ou `angles_cal` (calibré)
    fichier : si fourni (ex. "angle_brut.png"), sauvegarde la figure dans
              ce fichier en plus (ou à la place) de l'afficher à l'écran —
              indispensable si plt.show() n'ouvre aucune fenêtre (environnement
              distant / headless, cas fréquent sur Onyxia/JupyterHub).
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for k, couleur in enumerate(COULEURS):
        axes[k].plot(temps, angles[k], color=couleur)
        axes[k].set_title(TITRES[k], fontsize=10)
        axes[k].set_ylabel("deg")
        axes[k].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[-1].set_xlabel("Temps (s)")
    if titre_general:
        fig.suptitle(titre_general)
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


# %%
# ## 2. Effet de la calibration (retrait de la ROM)
#
# Superpose, pour chaque axe, la courbe brute et la courbe calibrée
# (angle brut − angle de référence ROM), pour visualiser l'écart introduit
# par la calibration. Affiche aussi la valeur de l'écart (angle de
# référence) directement sur chaque sous-graphique.

def afficher_effet_calibration(temps, angles, angles_cal, angles_ref, fichier=None):
    """
    temps       : array (n_frames,) en secondes
    angles      : array (3, n_frames) en degrés — angles bruts
    angles_cal  : array (3, n_frames) en degrés — angles calibrés
                  (= angles - angles_ref)
    angles_ref  : array (3,) en degrés — angle(s) de référence retiré(s)
                  (calculé sur l'essai ROM / ANAT_REF)
    fichier     : si fourni, sauvegarde la figure dans ce fichier (voir
                  afficher_angles)
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for k, couleur in enumerate(COULEURS):
        axes[k].plot(temps, angles[k], color=couleur, alpha=0.4, label="brut")
        axes[k].plot(temps, angles_cal[k], color=couleur, label="calibré")
        axes[k].axhline(angles_ref[k], color=couleur, linewidth=0.8,
                         linestyle=":", alpha=0.7,
                         label=f"référence ROM = {angles_ref[k]:.1f}°")
        axes[k].set_title(TITRES[k], fontsize=10)
        axes[k].set_ylabel("deg")
        axes[k].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle("Effet de la calibration (retrait de la ROM)")
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


# %%
# ## Utilisation
#
# Appels effectifs : à mettre après les définitions des deux fonctions,
# et après avoir calculé temps, angles, angles_cal, angles_ref (dans
# projet_chgmt_repere_corrige.py, ou copiés/collés juste avant ces lignes).
#
# Le paramètre `fichier` sauvegarde chaque figure en PNG dans le dossier
# courant : utile si plt.show() n'ouvre pas de fenêtre (terminal distant/
# headless, cas fréquent sur Onyxia). Ouvrez ensuite les .png depuis
# l'explorateur de fichiers de Jupyter/VS Code.

afficher_angles(temps, angles, "Angle du genou droit (brut)", fichier="angle_brut.png")
afficher_angles(temps, angles_cal, "Angle du genou droit (calibré)", fichier="angle_calibre.png")
afficher_effet_calibration(temps, angles, angles_cal, angles_ref, fichier="effet_calibration.png")
