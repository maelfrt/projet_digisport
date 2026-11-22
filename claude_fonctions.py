"""Digisport M1 - Influence de l'inclinaison du dos sur la frappe de football"""

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyomeca import Markers
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr

# %% Paramètres (à adapter si besoin)
DOSSIER = "/home/onyxia/work/projet_digisport/data2"

ESSAIS = {  # fichier : (position, numéro de capture)
    "arriere1_ball_precis.c3d": ("Arrière", 1), "arriere2_ball_precis.c3d": ("Arrière", 2),
    "droit1_ball_precis.c3d": ("Droit", 1),   "droit2_ball_precis.c3d": ("Droit", 2),
    "avant1_ball_precis.c3d": ("Avant", 1),   "avant2_ball_precis.c3d": ("Avant", 2),
}
ORDRE = ["Arrière", "Droit", "Avant"]
COULEURS = {"Arrière": "tab:red", "Droit": "tab:green", "Avant": "tab:blue"}

print("Dossier des essais :", DOSSIER)
print("Contenu du dossier :")

for fichier in os.listdir(DOSSIER):
    print(fichier)

REF_FILE = None  
CUTOFF_CORPS = 15        # Hz (corps). Le ballon n'est pas filtré.
SEUIL_IMPACT = 4.5      # m/s : vitesse du ballon qui définit l'impact
N_FRAMES_BALLON = 5      # nb de frames après impact pour la vitesse initiale du ballon
DIRECTION_BUT = None     # ex : (1, 0) si le but est dans le sens des X+ ; None = auto

BAS_TRONC = ["hanche droite", "hanche gauche"]
HAUT_TRONC = ["epaule droite", "epaule gauche"]
MARQUEURS_CORPS = sorted(set(BAS_TRONC + HAUT_TRONC + [
    "genou exterieur droit", "cheville droite", "pied droit", "pied gauche"]))


# %% Fonctions : traitement du signal
def interpoler_nan(x):
    """Interpole linéairement les NaN de chaque axe. x : (3, n)."""
    x = x.copy()
    idx = np.arange(x.shape[1])
    for i in range(3):
        ok = ~np.isnan(x[i])
        x[i] = np.interp(idx, idx[ok], x[i][ok])
    return x


def lowpass(x, cutoff, fs, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, x, axis=-1)


def charger_essai(chemin, cutoff=CUTOFF_CORPS):
    """Renvoie (marqueurs corps filtrés en m, ballon en m non filtré, fréquence)."""
    print("Chemin du fichier :", chemin)
    print("Fichier existant :", os.path.exists(chemin))
    print("Chemin absolu :", os.path.abspath(chemin))
    rep = Markers.from_c3d(chemin)
    freq = float(rep.attrs["rate"])
    noms = [str(n) for n in rep.channel.values]

    def get(nom):
        return np.array(rep.sel(channel=nom, axis=["x", "y", "z"]), dtype=float).reshape(3, -1)

    mk = {}
    for nom in MARQUEURS_CORPS:
        x = get(nom)
        n_nan = int(np.isnan(x[0]).sum())
        if n_nan > 0:
            print(f"   {nom} : {n_nan} frames interpolées")
        mk[nom] = lowpass(interpoler_nan(x), cutoff, freq) / 1000
    ballon = get("ballon_moyenne") / 1000 if "ballon_moyenne" in noms else None
    return mk, ballon, freq


def centre(mk, noms):
    return np.mean([mk[n] for n in noms], axis=0)


# %% Fonctions : ballon
def detecter_impact(ballon, freq, seuil=SEUIL_IMPACT):
    """Première frame où la vitesse du ballon dépasse le seuil (NaN ignorés)."""
    vit = np.linalg.norm(np.diff(ballon, axis=1), axis=0) * freq
    sup = np.where(vit > seuil)[0]
    return (int(sup[0]) if len(sup) else None), vit


def metriques_ballon(ballon, impact, freq, n_max=N_FRAMES_BALLON):
    """Vitesse initiale, élévation et vecteur vitesse (régression linéaire)."""
    idx = np.arange(impact + 1, min(impact + 1 + n_max, ballon.shape[1]))
    idx = idx[~np.isnan(ballon[:, idx]).any(axis=0)]
    if len(idx) < 2:
        return None
    t = (idx - idx[0]) / freq
    v = np.array([np.polyfit(t, ballon[i, idx], 1)[0] for i in range(3)])
    vitesse = np.linalg.norm(v)
    elevation = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
    return vitesse, elevation, v, len(idx)


def azimut_relatif(v, avant):
    """Écart horizontal par rapport à l'axe de tir (° ; > 0 = vers la gauche)."""
    gauche = np.array([-avant[1], avant[0]])
    return np.degrees(np.arctan2(v[0]*gauche[0] + v[1]*gauche[1],
                                 v[0]*avant[0] + v[1]*avant[1]))


# %% Fonctions : angles et cinématique
def calculer_angle(p1, p2, p3, p4):
    """Angle (°) entre p1->p2 et p3->p4 (tableaux (3, n)). 0° = segments alignés."""
    v1, v2 = p2 - p1, p4 - p3
    v1 = v1 / np.linalg.norm(v1, axis=0)
    v2 = v2 / np.linalg.norm(v2, axis=0)
    return np.degrees(np.arccos(np.clip(np.sum(v1 * v2, axis=0), -1.0, 1.0)))


def inclinaison_tronc(bassin, epaule, avant):
    """Angle signé du tronc / verticale Z (°). > 0 : penché vers l'avant (cible)."""
    v = epaule - bassin
    return np.degrees(np.arctan2(v[0]*avant[0] + v[1]*avant[1], v[2]))


def rotation_bassin(hanche_d, hanche_g, avant):
    """Rotation horizontale du bassin (°). 0 = bassin face à la cible ;
    > 0 = hanche droite en avant de la gauche."""
    v = hanche_g - hanche_d
    gauche = np.array([-avant[1], avant[0]])
    return np.degrees(np.arctan2(-(v[0]*avant[0] + v[1]*avant[1]),
                                 v[0]*gauche[0] + v[1]*gauche[1]))


def moyenne_fenetre(signal, imp, freq, duree=0.1):
    """Moyenne du signal sur [impact - duree ; impact]."""
    n = int(duree * freq)
    return float(np.mean(signal[max(0, imp - n):imp + 1]))


def vitesse_marqueur(m, freq):
    """Vitesse (m/s) d'un marqueur. m : (3, n) en m."""
    return np.linalg.norm(np.gradient(m, axis=1), axis=0) * freq


def vitesse_pied_impact(pied, imp, freq, fenetre=5):
    """Pic de vitesse du pied sur les `fenetre` frames avant l'impact (m/s)."""
    v = vitesse_marqueur(pied, freq)
    return float(v[max(0, imp - fenetre):imp + 1].max())


def position_pied_appui(pied_appui, ballon, imp, avant, n_ref=10):
    """Position du pied d'appui par rapport au ballon à l'impact (m).
    Position du ballon = médiane sur les n_ref frames avant l'impact.
    Renvoie (antéro-postérieur, latéral) :
      AP  > 0 : pied d'appui en avant du ballon, < 0 : derrière
      LAT > 0 : à gauche de l'axe de tir, < 0 : à droite"""
    ref = np.nanmedian(ballon[:, max(0, imp - n_ref):imp + 1], axis=1)
    if np.isnan(ref).any():
        return np.nan, np.nan
    d = pied_appui[:, imp] - ref
    gauche = np.array([-avant[1], avant[0]])
    return float(d[0]*avant[0] + d[1]*avant[1]), float(d[0]*gauche[0] + d[1]*gauche[1])


def vitesse_angulaire(angle, freq):
    """Dérivée d'un angle (°/s)."""
    return np.gradient(angle, 1 / freq)


def pic_extension_genou(angle_genou, imp, freq, fenetre_s=0.2):
    """Pic de vitesse angulaire d'extension du genou sur [impact - 0,2 s ; impact] (°/s).
    L'angle augmente avec la flexion : l'extension est une vitesse négative."""
    w = vitesse_angulaire(angle_genou, freq)[max(0, imp - int(fenetre_s * freq)):imp + 1]
    return float(-w.min())


def rapport_transfert(v_ballon, v_pied):
    """Rapport vitesse ballon / vitesse pied (sans unité)."""
    return float(v_ballon / v_pied) if v_pied > 0 else np.nan


# %% Passe 1 : chargement, impact, vitesse initiale du ballon
essais = {}
for fichier, (pos, cap) in ESSAIS.items():
    print(f"\n{fichier} ({pos}, capture {cap})")
    mk, ballon, freq = charger_essai(os.path.join(DOSSIER, fichier))
    if ballon is None:
        print("   marqueur 'ballon_moyenne' absent, essai ignoré")
        continue
    valides = np.where(~np.isnan(ballon[0]))[0]
    if len(valides) == 0:
        print("   ballon jamais suivi, essai ignoré")
        continue
    print(f"   ballon suivi de la frame {valides[0]} à {valides[-1]} "
          f"({len(valides)} valides sur {ballon.shape[1]})")
    imp, vit = detecter_impact(ballon, freq)
    if imp is None:
        print("   IMPACT NON DÉTECTÉ (baisse SEUIL_IMPACT ou vérifie le ballon)")
        continue
    avant_imp = vit[max(0, imp - 20):imp]
    if np.sum(~np.isnan(avant_imp)) > 0 and np.nanmedian(avant_imp) > 0.3:
        print("   ATTENTION : le ballon bouge avant l'impact détecté, vérifie la figure de contrôle")
    res = metriques_ballon(ballon, imp, freq)
    if res is None:
        print("   moins de 2 frames de ballon après l'impact, essai ignoré")
        continue
    vitesse, elevation, vvec, n_ok = res
    print(f"   impact frame {imp} | {n_ok} frames de ballon utilisées | "
          f"v = {vitesse:.1f} m/s | élévation = {elevation:.1f}°")
    essais[fichier] = {"pos": pos, "cap": cap, "mk": mk, "ballon": ballon, "freq": freq, "imp": imp,
    "vit": vit, "vitesse": vitesse, "elevation": elevation, "vvec": vvec, "n_ok": n_ok}

# %% Figure de contrôle : vitesse du ballon et impact détecté
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
for ax, (fichier, d) in zip(axes.flat, essais.items()):
    t = np.arange(len(d["vit"])) / d["freq"]
    ax.plot(t, d["vit"], ".-")
    ax.plot(t[d["imp"]], d["vit"][d["imp"]], "rx", ms=12, mew=2)
    ax.axhline(SEUIL_IMPACT, color="gray", ls=":")
    NOMS_ESSAIS = {
    "droit2_ball_precis.c3d": "Droit C2",
    "droit1_ball_precis.c3d": "Droit C1",
    "arriere1_ball_precis.c3d": "Arrière C1",
    "arriere2_ball_precis.c3d": "Arrière C2",
    "avant2_ball_precis.c3d": "Avant C2",
    "avant1_ball_precis.c3d": "Avant C1"
    } 

    ax.set_title(NOMS_ESSAIS.get(fichier, fichier)) 
    ax.set_xlabel("Temps (s)")
axes[0, 0].set_ylabel("Vitesse ballon (m/s)")
axes[1, 0].set_ylabel("Vitesse ballon (m/s)")
fig.suptitle("Détection du moment du tir ")
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "fig_controle_impact.png"), dpi=150)
plt.show()

# %% Axe de tir commun aux essais
if DIRECTION_BUT is not None:
    avant = np.array(DIRECTION_BUT, dtype=float)
else:
    dirs = [d["vvec"][:2] / np.linalg.norm(d["vvec"][:2]) for d in essais.values()]
    avant = np.mean(dirs, axis=0)
avant = avant / np.linalg.norm(avant)
print("\nAxe de tir (x, y) :", np.round(avant, 3))

# %% Référence anatomique (0° du tronc), optionnelle
# %% Référence anatomique : inclinaison moyenne du tronc en position debout
REF_FILE = "ANAT_REF.c3d"

mk_ref, _, freq_ref = charger_essai(os.path.join(DOSSIER, REF_FILE))
bassin_ref = centre(mk_ref, BAS_TRONC)
epaule_ref = centre(mk_ref, HAUT_TRONC)

# 1) Direction de face du sujet dans la capture de référence
#    (perpendiculaire à l'axe hanche droite -> hanche gauche, dans le plan horizontal)
v_hanches = np.mean(mk_ref["hanche gauche"] - mk_ref["hanche droite"], axis=1)
gauche_ref = v_hanches[:2] / np.linalg.norm(v_hanches[:2])
face_ref = np.array([gauche_ref[1], -gauche_ref[0]])      # même convention que 'avant'

# 2) Inclinaison au cours de la capture, projetée sur deux axes :
#    - face_ref : direction réelle de face du sujet (référence anatomique)
#    - avant    : axe de tir utilisé dans les essais
incl_face = inclinaison_tronc(bassin_ref, epaule_ref, face_ref)
incl_tir = inclinaison_tronc(bassin_ref, epaule_ref, avant)

# 3) Ne garder que la phase où le sujet est immobile (le début et la fin
#    de la capture contiennent souvent des mouvements)
vit_tronc = np.linalg.norm(np.gradient(epaule_ref, axis=1), axis=0) * freq_ref   # m/s
stable = vit_tronc < 0.05                                                         # < 5 cm/s

print(f"Capture de référence : {len(incl_face)} images, {stable.sum()} images stables")
print(f"Angle entre la face du sujet et l'axe de tir : "
      f"{np.degrees(np.arccos(np.clip(face_ref @ avant, -1, 1))):.1f}°")
print(f"Inclinaison de face (moyenne ± écart-type) : "
      f"{incl_face[stable].mean():.1f} ± {incl_face[stable].std():.1f}°")
print(f"Inclinaison projetée sur l'axe de tir      : "
      f"{incl_tir[stable].mean():.1f} ± {incl_tir[stable].std():.1f}°")

INCL_REF = float(np.median(incl_face[stable]))
print(f"\nINCL_REF retenu : {INCL_REF:.1f}°")

# Contrôle visuel
t_ref = np.arange(len(incl_face)) / freq_ref
plt.figure(figsize=(8, 3.5))
plt.plot(t_ref, incl_face, label="inclinaison (face du sujet)")
plt.plot(t_ref, incl_tir, label="inclinaison (axe de tir)", alpha=0.6)
plt.scatter(t_ref[stable], incl_face[stable], s=6, color="k", label="images stables")
plt.axhline(INCL_REF, color="r", ls="--")
plt.xlabel("Temps (s)"); plt.ylabel("Inclinaison (°)"); plt.legend()
plt.title("Position anatomique de référence")
plt.savefig(os.path.join(DOSSIER, "fig_reference_anatomique.png"), dpi=150)
plt.show()


INCL_REF = 0.0
if REF_FILE:
    mk_ref, _, _ = charger_essai(os.path.join(DOSSIER, REF_FILE))
    INCL_REF = float(np.mean(inclinaison_tronc(centre(mk_ref, BAS_TRONC),
    centre(mk_ref, HAUT_TRONC), avant)))
    print(f"Inclinaison de référence (position anatomique) : {INCL_REF:.1f}°")
else:
    print("Pas de référence : inclinaison par rapport à la verticale (0° = tronc vertical).")

# %% Passe 2 : variables biomécaniques par essai (priorités 1 et 2)
lignes, courbes = [], {}
for fichier, d in essais.items():
    mk, freq, imp, ballon = d["mk"], d["freq"], d["imp"], d["ballon"]
    n = mk["pied droit"].shape[1]
    bassin, epaule = centre(mk, BAS_TRONC), centre(mk, HAUT_TRONC)

    # Signaux temporels
    incl = inclinaison_tronc(bassin, epaule, avant) - INCL_REF
    rot = rotation_bassin(mk["hanche droite"], mk["hanche gauche"], avant)
    genou = calculer_angle(mk["hanche droite"], mk["genou exterieur droit"],
                           mk["genou exterieur droit"], mk["cheville droite"])

    # Priorité 1
    v_pied = vitesse_pied_impact(mk["pied droit"], imp, freq)

    # Priorité 2
    ap, lat = position_pied_appui(mk["pied gauche"], ballon, imp, avant)

    lignes.append({
        "position": d["pos"], "capture": d["cap"], "fichier": fichier,
        "impact_frame": imp, "n_frames_ballon": d["n_ok"],
        # priorité 1
        "incl_impact": incl[imp],
        "incl_moy_100ms": moyenne_fenetre(incl, imp, freq),
        "v_ballon": d["vitesse"],
        "elevation": d["elevation"],
        "v_pied": v_pied,
        # priorité 2
        "appui_AP_cm": ap * 100,
        "appui_lat_cm": lat * 100,
        "angle_genou_impact": genou[imp],
        "vang_ext_genou": pic_extension_genou(genou, imp, freq),
        "rot_bassin_impact": rot[imp],
        "rapport_v_ballon_pied": rapport_transfert(d["vitesse"], v_pied),
        # info
        "azimut_rel": azimut_relatif(d["vvec"], avant),
    })

    a, b = max(0, imp - int(0.5 * freq)), min(n, imp + int(0.2 * freq) + 1)
    t = (np.arange(a, b) - imp) / freq
    courbes[fichier] = dict(t=t, incl=incl[a:b], rot=rot[a:b], genou=genou[a:b])

df = pd.DataFrame(lignes)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n", df.round(1).to_string(index=False))
df.to_csv(os.path.join(DOSSIER, "resultats_digisport.csv"), sep=";", decimal=",", index=False)

# %% Statistiques descriptives et corrélations exploratoires
cols = ["incl_impact", "incl_moy_100ms", "v_ballon", "elevation", "v_pied",
        "appui_AP_cm", "appui_lat_cm", "angle_genou_impact", "vang_ext_genou",
        "rot_bassin_impact", "rapport_v_ballon_pied"]
print("\n--- Moyenne / min / max par position ---")
for c in cols:
    g = df.groupby("position")[c].agg(["mean", "min", "max"]).reindex(ORDRE).round(1)
    print(f"\n{c}\n{g}")

paires = [("incl_impact", "elevation"), ("incl_impact", "v_ballon"),
          ("incl_impact", "v_pied"), ("incl_impact", "appui_AP_cm"),
          ("appui_AP_cm", "elevation"), ("v_pied", "v_ballon"),
          ("incl_impact", "rapport_v_ballon_pied")]

# %% Figure : variables clés par position (points individuels + moyenne)

variables_fig = [
        "incl_impact", "v_ballon", "elevation", "v_pied",
        "appui_AP_cm", "appui_lat_cm", "angle_genou_impact",
        "vang_ext_genou", "rot_bassin_impact", "rapport_v_ballon_pied"
    ]

# Correspondance entre les noms des variables et les titres des graphiques
NOMS_VARIABLES = {
    "incl_impact": "Inclinaison du tronc à l'impact (°)",
    "v_ballon": "Vitesse initiale du ballon (m/s)",
    "elevation": "Angle d'élévation du ballon (°)",
    "v_pied": "Vitesse maximale du pied avant l'impact (m/s)",
    "appui_AP_cm": "Position avant-arrière du pied d'appui (cm)",
    "appui_lat_cm": "Position latérale du pied d'appui (cm)",
    "angle_genou_impact": "Angle du genou droit à l'impact (°)",
    "vang_ext_genou": "Vitesse angulaire d'extension du genou (°/s)",
    "rot_bassin_impact": "Rotation du bassin à l'impact (°)",
    "rapport_v_ballon_pied": "Rapport vitesse ballon / vitesse pied"
    }

fig, axes = plt.subplots(3, 4, figsize=(20, 12))

for ax, var in zip(axes.flat, variables_fig):

        sns.stripplot(
            x="position",
            y=var,
            data=df,
            order=ORDRE,
            ax=ax,
            color="black",
            size=9
        )

    # Calcul de la moyenne pour chaque position
        m = df.groupby("position")[var].mean().reindex(ORDRE)

    # Affichage de la moyenne en rouge
        ax.scatter(
            range(len(ORDRE)),
            m.values,
            marker="_",
            s=900,
            color="red",
            zorder=3,
            label="Moyenne"
        )

        ax.set_xlabel("Position du tronc")
        ax.set_ylabel("")
        ax.set_title(NOMS_VARIABLES[var], fontsize=10)

for ax in axes.flat[len(variables_fig):]:
    ax.axis("off")

    fig.suptitle(
        "Comparaison des paramètres biomécaniques selon la position du tronc",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER, "fig_variables_par_position.png"), dpi=150)
    plt.show()

    m = df.groupby("position")[var].mean().reindex(ORDRE)
    ax.scatter(range(len(ORDRE)), m.values, marker="_", s=900, color="red", zorder=3)
    ax.set_xlabel("Position du dos")
    ax.set_title(var)
for ax in axes.flat[len(variables_fig):]:
    ax.axis("off")
fig.suptitle("Impact de l'inclinaison sur différents paramettres")
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "fig_variables_par_position.png"), dpi=150)
plt.show()

# %% Figure : nuages de points inclinaison mesurée vs résultat
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (x, y) in zip(axes, [("incl_impact", "elevation"), ("incl_impact", "v_ballon"),
                              ("incl_impact", "v_pied")]):
    for pos in ORDRE:
        s = df[df["position"] == pos]
        ax.scatter(s[x], s[y], s=80, color=COULEURS[pos], label=pos)
        for _, r in s.iterrows():
            ax.annotate(f"C{int(r['capture'])}", (r[x], r[y]),
                        textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Inclinaison du tronc à l'impact (°)")
    ax.set_ylabel(y)
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "fig_nuages.png"), dpi=150)
plt.show()

# %% Figure : courbes recalées sur l'impact
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 11))
for fichier, c in courbes.items():
    pos, cap = ESSAIS[fichier]
    style = "-" if cap == 1 else "--"
    axes[0].plot(c["t"], c["incl"], style, color=COULEURS[pos], label=f"{pos} C{cap}")
    axes[1].plot(c["t"], c["rot"], style, color=COULEURS[pos])
    axes[2].plot(c["t"], c["genou"], style, color=COULEURS[pos])
for ax in axes:
    ax.axvline(0, color="k", lw=1)
axes[0].set_ylabel("Inclinaison du tronc (°)")
axes[1].set_ylabel("Rotation du bassin (°)")
axes[2].set_ylabel("Angle du genou droit (°)")
axes[2].set_xlabel("Temps par rapport à l'impact (s)")
axes[0].legend(ncol=3)
fig.suptitle("Cinématique autour de l'impact (trait plein = capture 1, tirets = capture 2)")
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "fig_courbes_impact.png"), dpi=150)
plt.show()

# %% Sensibilité au filtre (pour justifier le cutoff dans le rapport)
print("\n--- Vitesse du pied (m/s) selon le cutoff ---")
for cutoff in (10, 15, 20):
    vals = []
    for fichier, d in essais.items():
        mk_c, _, fr = charger_essai(os.path.join(DOSSIER, fichier), cutoff=cutoff)
        vals.append(vitesse_pied_impact(mk_c["pied droit"], d["imp"], fr))
    print(f"{cutoff} Hz :", np.round(vals, 1))