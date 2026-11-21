
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.spatial.transform import Rotation

plt.style.use("default")

DOSSIER = "projet_digisport/data2/"
FICHIER_REF = DOSSIER + "ANAT_REF.c3d"

CAPTURES = {
    "Arrière": ["CAPTURE0002_ballon_v5.c3d", "CAPTURE0004_ballon_v5.c3d"],
    "Neutre":  ["CAPTURE0005_ballon_v5.c3d", "CAPTURE0007_ballon_v5.c3d"],
    "Avant":   ["CAPTURE0008_ballon_v5.c3d", "CAPTURE0009_ballon_v5.c3d"],
}
COULEURS = {"Arrière": "tab:red", "Neutre": "tab:green", "Avant": "tab:blue"}
NOMS_ANGLES = ["Flexion (+) / extension (-)",
               "Abduction (+) / adduction (-)",
               "Rotation interne (+) / externe (-)"]

# --- Détection de la frappe ---
MARQUEURS_PIED = ["pied droit"]                 # noms essayés dans l'ordre
MARQUEURS_BALLON = ["ballon_moyenne", "ballon"]  # noms essayés dans l'ordre
SEUIL_BALLON_MM = 50    # le ballon est considéré "parti" s'il s'éloigne de plus de 50 mm
                        # de sa position de départ

# --- Fenêtre analysée autour de la frappe ---
DUREE_APRES_S = 0.0     # secondes conservées APRÈS la frappe (0 = on s'arrête à l'impact)
TROU_MAX_S = 0.05       # un trou de données plus long que ça n'est pas comblé


# %%
# ## Fonctions de base (identiques à votre script qui fonctionne)

def position(essai, nom):
    return essai.sel(channel=nom).values[0:3, :]  # (3, n_frames), en mm


def normaliser(v):
    norme = np.linalg.norm(v)
    if norme < 1e-8:
        raise ValueError("Vecteur quasi nul")
    return v / norme


def repere_cuisse(cuisse, gnext, gnint):
    origine = (gnext + gnint) / 2
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(cuisse - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


def repere_jambe(gnext, gnint, mollet, cheville):
    origine = cheville
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(mollet - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


NOMS_MARQUEURS = ["cuisse droite", "genou exterieur droit", "genou interieur droite",
                  "mollet droite", "cheville droite"]


def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [position(essai, nom) for nom in NOMS_MARQUEURS]
    n_frames = cuisse.shape[1]
    angles = np.full((3, n_frames), np.nan)
    for i in range(n_frames):
        pts = [m[:, i] for m in (cuisse, gnext, gnint, mollet, cheville)]
        if any(np.isnan(p).any() for p in pts):
            continue
        try:
            R_rel = repere_cuisse(pts[0], pts[1], pts[2]).T @ repere_jambe(pts[1], pts[2], pts[3], pts[4])
            angles[:, i] = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
        except ValueError:
            continue
    return angles


def angle_reference(fichier_ref):
    marqueurs_ref = Markers.from_c3d(fichier_ref)
    angles_ref_tous = angles_genou_droit(marqueurs_ref)
    cheville_ref = position(marqueurs_ref, "cheville droite")
    vitesse = np.convolve(np.linalg.norm(np.diff(cheville_ref, axis=1), axis=0),
                          np.ones(50) / 50, mode="same")
    return angles_ref_tous[:, np.nanargmin(vitesse)]


# %%
# ## Détection de l'instant de frappe
#
# - d(t) = distance entre le marqueur du pied et celui du ballon.
# - Le ballon est immobile avant la frappe : on repère la 1re image où il
#   s'est déplacé de plus de SEUIL_BALLON_MM par rapport à sa position de
#   départ ("i_parti").
# - L'impact est l'image où d(t) est minimale, en ne regardant que AVANT
#   (et jusqu'à) i_parti. Ça évite de prendre un moment où le pied repasse
#   près du ballon après la frappe, ou où le joueur est déjà à côté du
#   ballon au début.

def choisir_marqueur(marqueurs, candidats, role):
    """Premier nom de `candidats` présent dans la capture, sinon erreur claire."""
    presents = marqueurs.channel.values.tolist()
    for nom in candidats:
        if nom in presents:
            return nom
    raise KeyError(f"Aucun marqueur '{role}' trouvé parmi {candidats}. "
                   f"Marqueurs de cette capture : {presents}")


def detecter_impact(marqueurs):
    nom_pied = choisir_marqueur(marqueurs, MARQUEURS_PIED, "pied")
    nom_ballon = choisir_marqueur(marqueurs, MARQUEURS_BALLON, "ballon")
    print(f"  marqueurs utilisés pour l'impact : {nom_pied} / {nom_ballon}")
    pied = position(marqueurs, nom_pied)
    ballon = position(marqueurs, nom_ballon)
    distance = np.linalg.norm(pied - ballon, axis=0)       # NaN si marqueur manquant

    images_ballon = np.where(~np.isnan(ballon).any(axis=0))[0]
    if len(images_ballon) == 0:
        raise ValueError("Marqueur du ballon absent sur toute la capture")
    depart = np.median(ballon[:, images_ballon[:20]], axis=1)
    deplacement = np.linalg.norm(ballon - depart[:, None], axis=0)
    parti = np.where(deplacement > SEUIL_BALLON_MM)[0]

    if len(parti) > 0:
        i_parti = int(parti[0])
        fenetre = distance[: i_parti + 1]
    else:
        i_parti = None
        print("  ATTENTION : le ballon ne bouge pas de plus de "
              f"{SEUIL_BALLON_MM} mm, on cherche le minimum sur toute la capture")
        fenetre = distance

    if np.all(np.isnan(fenetre)):
        raise ValueError("Distance pied-ballon inconnue avant le départ du ballon")
    return int(np.nanargmin(fenetre)), i_parti, distance


# %%
# ## Mise sur une grille de temps commune (secondes, 0 = impact)

def sur_grille(t, y, grille, trou_max_s=TROU_MAX_S):
    valides = np.where(~np.isnan(y))[0]
    if len(valides) < 2:
        return np.full(len(grille), np.nan)
    tv = t[valides]
    res = np.interp(grille, tv, y[valides])

    pos = np.searchsorted(tv, grille)
    gauche = tv[np.clip(pos - 1, 0, len(tv) - 1)]
    droite = tv[np.clip(pos, 0, len(tv) - 1)]
    trou = ((droite - gauche) > trou_max_s) & ~np.isclose(grille, droite)
    res[trou] = np.nan
    res[(grille < tv[0] - 1e-9) | (grille > tv[-1] + 1e-9)] = np.nan
    return res


def moyenne_captures(essais):
    pile = np.stack(essais)                       # (n_essais, 3, n_points)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(pile, axis=0)


# %%
# ## Traitement de toutes les captures

def traiter_tout():
    angles_ref = angle_reference(FICHIER_REF)
    print("angles de référence (deg) :", np.round(angles_ref, 1))

    captures = []
    for mouvement, fichiers in CAPTURES.items():
        for f in fichiers:
            marqueurs = Markers.from_c3d(DOSSIER + f)
            rate = float(marqueurs.attrs["rate"])
            n = marqueurs.shape[2]
            angles_cal = angles_genou_droit(marqueurs) - angles_ref[:, None]
            i_imp, i_parti, distance = detecter_impact(marqueurs)
            captures.append(dict(mouvement=mouvement, fichier=f, rate=rate, n=n,
                                 angles=angles_cal, i_imp=i_imp, i_parti=i_parti,
                                 distance=distance))
            print(f"{mouvement:8s} {f} : {n/rate:.2f} s, impact à t = {i_imp/rate:.2f} s "
                  f"(image {i_imp}), {i_imp/rate:.2f} s avant / {(n-1-i_imp)/rate:.2f} s après")

    # durée avant l'impact = la plus courte des 6 captures
    limitante = min(captures, key=lambda c: c["i_imp"] / c["rate"])
    duree_avant = limitante["i_imp"] / limitante["rate"]
    print(f"\nDurée avant impact retenue : {duree_avant:.2f} s "
          f"(limitée par {limitante['fichier']})")

    rate = max(c["rate"] for c in captures)
    nb_avant = int(np.floor(duree_avant * rate))
    nb_apres = int(np.floor(DUREE_APRES_S * rate))
    grille = np.arange(-nb_avant, nb_apres + 1) / rate

    essais_par_mouvement = {m: [] for m in CAPTURES}
    for c in captures:
        t_rel = (np.arange(c["n"]) - c["i_imp"]) / c["rate"]
        c["sur_grille"] = np.array([sur_grille(t_rel, a, grille) for a in c["angles"]])
        essais_par_mouvement[c["mouvement"]].append(c["sur_grille"])

    moyennes = {m: moyenne_captures(e) for m, e in essais_par_mouvement.items()}
    return grille, essais_par_mouvement, moyennes, captures


# %%
# ## Graphique principal : une courbe moyenne par mouvement, alignée sur la frappe

def tracer_moyennes(grille, essais_par_mouvement, moyennes, composante=0, fichier=None):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for mouvement, couleur in COULEURS.items():
        for essai in essais_par_mouvement[mouvement]:
            ax.plot(grille, essai[composante], color=couleur, alpha=0.25, linewidth=1)
        ax.plot(grille, moyennes[mouvement][composante], color=couleur,
                linewidth=2.5, label=f"{mouvement} (moyenne)")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.text(0, ax.get_ylim()[1], " frappe", va="top", ha="left", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title(f"Genou droit : {NOMS_ANGLES[composante]}")
    ax.set_xlabel("Temps par rapport à la frappe (s)")
    ax.set_ylabel("deg")
    ax.legend()
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150, facecolor="white", transparent=False)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


# %%
# ## Figure de contrôle : l'impact est-il bien détecté ?
#
# Pour chaque capture : distance pied-ballon, impact détecté (trait noir),
# moment où le ballon part (pointillé orange) et fenêtre conservée (zone grise).

def tracer_controle_impacts(captures, grille, fichier=None):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), facecolor="white", sharey=True)
    for ax, c in zip(axes.ravel(), captures):
        t = np.arange(c["n"]) / c["rate"]
        t_imp = c["i_imp"] / c["rate"]
        ax.plot(t, c["distance"], color=COULEURS[c["mouvement"]])
        ax.axvspan(t_imp + grille[0], t_imp + grille[-1], color="grey", alpha=0.2)
        ax.axvline(t_imp, color="black", linewidth=1.2)
        if c["i_parti"] is not None:
            ax.axvline(c["i_parti"] / c["rate"], color="orange", linestyle=":", linewidth=1.2)
        ax.set_title(f"{c['mouvement']} - {c['fichier'][:11]}", fontsize=9)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("distance pied-ballon (mm)")
    fig.suptitle("Contrôle : noir = impact détecté, orange = ballon parti, gris = fenêtre conservée")
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150, facecolor="white", transparent=False)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


# %%
if __name__ == "__main__":
    grille, essais_par_mouvement, moyennes, captures = traiter_tout()
    tracer_controle_impacts(captures, grille, fichier="controle_impacts.png")
    tracer_moyennes(grille, essais_par_mouvement, moyennes, composante=0,
                    fichier="moyenne_flexion_alignee_frappe.png")