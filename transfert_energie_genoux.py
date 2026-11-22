# -*- coding: utf-8 -*-

# %%
# # Inclinaison du tronc, transfert pied -> ballon et vitesse angulaire du genou
#
# Partie 1 - Transfert pied -> ballon
#   Pour chaque capture : vitesse du pied juste avant l'impact, vitesse initiale
#   du ballon juste après, rapport des deux (efficacité du transfert) et
#   énergie cinétique du ballon. Puis lien avec l'inclinaison du tronc.
#
# Partie 2 - Vitesse angulaire du genou droit
#   Angle du genou calculé dans des repères locaux cuisse / jambe (comme dans
#   le TD "changement de repère" et le premier script), calibré avec ANAT_REF,
#   puis dérivé par rapport au temps. Puis lien avec l'inclinaison du tronc.
#
# Les marqueurs du corps viennent des fichiers "*_pas_ball.c3d", ceux du
# ballon des fichiers "*_ball_precis.c3d" (même capture, même axe du temps).

# %%
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata

plt.style.use("default")

# ---------------------------------------------------------------- Données
# Les chemins sont construits à partir du dossier où se trouve ce script.
# On suppose que le dossier "data2" est situé à côté de ce fichier Python.
BASE_DIR = Path(__file__).resolve().parent
DOSSIER = BASE_DIR / "data2"
FICHIER_REF = DOSSIER / "ANAT_REF.c3d"


def charger_c3d(chemin, description="fichier C3D"):
    """Vérifie l'existence du fichier puis le charge avec pyomeca."""
    chemin = Path(chemin)

    if not chemin.is_file():
        raise FileNotFoundError(
            f"{description} introuvable : {chemin}\n"
            f"Dossier du script : {BASE_DIR}\n"
            f"Dossier de données attendu : {DOSSIER}\n"
            "Vérifiez que le dossier 'data2' est bien placé à côté du script "
            "et que le nom du fichier est exactement le bon."
        )

    return Markers.from_c3d(str(chemin))

# (fichier corps, fichier ballon) pour chaque capture. A ADAPTER si besoin.
CAPTURES = {
    "Arrière": [("arriere1_pas_ball.c3d", "arriere1_ball_precis.c3d"),
                ("arriere2_pas_ball.c3d", "arriere2_ball_precis.c3d")],
    "Neutre":  [("droit1_pas_ball.c3d", "droit1_ball_precis.c3d"),
                ("droit2_pas_ball.c3d", "droit2_ball_precis.c3d")],
    "Avant":   [("avant1_pas_ball.c3d", "avant1_ball_precis.c3d"),
                ("avant2_pas_ball.c3d", "avant2_ball_precis.c3d")],
}
COULEURS = {"Arrière": "tab:red", "Neutre": "tab:green", "Avant": "tab:blue"}

# ---------------------------------------------------------------- Réglages
MARQUEUR_PIED = "pied droit"      # pied qui frappe
SEUIL_BALLON_MM = 50              # le ballon est "parti" s'il s'éloigne de plus de 50 mm
AXE_VERTICAL = None               # None = détection automatique ; sinon 0, 1 ou 2 (x, y, z)
MASSE_BALLON_KG = 0.43            # ballon de football taille 5
MASSE_EFFECTIVE_PIED_KG = None    # facultatif (ex. 3.5 pour pied + jambe) : active le calcul du rendement
DUREE_VITESSE_BALLON_S = 0.06     # le ballon est suivi sur 0,06 s après son départ
FENETRE_VITESSE_PIED_S = 0.05     # vitesse du pied = maximum sur les 0,05 s avant l'impact
FENETRE_OMEGA_S = 0.30            # pic de vitesse angulaire cherché dans les 0,30 s avant l'impact
FENETRE_IMPACT_S = 0.05           # "valeur à l'impact" = moyenne sur les 0,05 s avant l'impact
TROU_MAX_S = 0.05                 # un trou de données plus court que ça est comblé
FENETRE_AVANT_S, FENETRE_APRES_S = 0.6, 0.1   # fenêtre des courbes (autour de l'impact)
R_CRITIQUE = 0.81                 # |r| de Pearson minimal pour être significatif (p < 0,05) avec 6 points
# Valeurs critiques du rho de Spearman (test bilatéral, p < 0,05) selon le nombre de captures valides.
# Avec 4 captures ou moins, aucune corrélation ne peut être significative.
SEUILS_SPEARMAN = {5: 1.0, 6: 0.886, 7: 0.786, 8: 0.738}


def seuil_spearman(n):
    return SEUILS_SPEARMAN.get(n, 0.7 if n > 8 else np.inf)


# %%
# ## Fonctions de base

_deja_signales = set()


def position(essai, nom):
    """(3, n_frames) en mm. Doublons de marqueurs : moyenne (et on le signale)."""
    v = essai.sel(channel=nom).values
    if v.ndim == 3:
        if (id(essai), nom) not in _deja_signales:
            _deja_signales.add((id(essai), nom))
            print(f"  ATTENTION : le marqueur '{nom}' existe {v.shape[1]} fois dans ce "
                  "fichier, moyenne des doublons utilisée")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(v[0:3], axis=1)
    return v[0:3, :]


def milieu(a, b):
    return (a + b) / 2


def moyenne(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a)


def combler_petits_trous(y, nmax):
    """Interpolation linéaire des trous de NaN d'au plus nmax images (entourés de données)."""
    y = y.copy()
    nan = np.isnan(y)
    i, n = 0, len(y)
    while i < n:
        if nan[i]:
            j = i
            while j < n and nan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= nmax:
                y[i:j] = np.interp(np.arange(i, j), [i - 1, j], [y[i - 1], y[j]])
            i = j
        else:
            i += 1
    return y


def derivee(y, rate, fenetre_s=0.07, ordre=3, angle=False):
    """Dérivée par rapport au temps (unité/s) par filtre de Savitzky-Golay, calculée
    sur chaque plage continue de données (les grands trous restent NaN)."""
    fen = max(int(round(fenetre_s * rate)) | 1, 5)                # impair, au moins 5
    y = combler_petits_trous(y, int(round(TROU_MAX_S * rate)))
    res = np.full(len(y), np.nan)
    bornes = np.flatnonzero(np.diff(np.r_[0, (~np.isnan(y)).astype(int), 0]))
    for deb, fin in zip(bornes[::2], bornes[1::2]):
        if fin - deb >= fen:
            seg = y[deb:fin]
            if angle:
                seg = np.degrees(np.unwrap(np.radians(seg)))
            res[deb:fin] = savgol_filter(seg, fen, ordre, deriv=1, delta=1 / rate)
    return res


def vitesse_marqueur(p, rate):
    """Vitesse (m/s) d'un marqueur (3, n) en mm."""
    return np.sqrt(sum(derivee(p[k], rate) ** 2 for k in range(3))) / 1000


def sur_grille(t, y, grille, trou_max_s=TROU_MAX_S):
    valides = np.where(~np.isnan(y))[0]
    if len(valides) < 2:
        return np.full(len(grille), np.nan)
    tv = t[valides]
    res = np.interp(grille, tv, y[valides])
    pos = np.searchsorted(tv, grille)
    gauche = tv[np.clip(pos - 1, 0, len(tv) - 1)]
    droite = tv[np.clip(pos, 0, len(tv) - 1)]
    res[((droite - gauche) > trou_max_s) & ~np.isclose(grille, droite)] = np.nan
    res[(grille < tv[0] - 1e-9) | (grille > tv[-1] + 1e-9)] = np.nan
    return res


# %%
# ## Ballon : départ, impact, vitesse initiale
#
# Le ballon vient du fichier "*_ball_precis". On suppose les deux fichiers
# synchrones (même capture) : l'impact est l'image où le pied (fichier corps)
# est le plus proche du ballon, juste avant que le ballon ne parte.

def positions_ballon(essai):
    """Marqueurs du ballon : tableau (n_marqueurs, 3, n_frames)."""
    noms = list(dict.fromkeys(str(n) for n in essai.channel.values.tolist()))
    ballons = [n for n in noms if "ball" in n.lower() and "moyenne" not in n.lower()]
    ballons = ballons or [n for n in noms if "ball" in n.lower()]
    if not ballons and len(noms) == 1:
        ballons = noms
    if not ballons:
        raise KeyError(f"Aucun marqueur de ballon dans ce fichier. Marqueurs trouvés : {noms}")
    print(f"  marqueurs ballon utilisés : {ballons}")
    return np.stack([position(essai, n) for n in ballons])


def depart_ballon(ballons):
    """Première image où le ballon s'est éloigné de plus de SEUIL_BALLON_MM de sa position de départ."""
    deplacements = []
    for b in ballons:
        valides = np.where(~np.isnan(b).any(axis=0))[0]
        if len(valides):
            depart = np.median(b[:, valides[:20]], axis=1)
            deplacements.append(np.linalg.norm(b - depart[:, None], axis=0))
    if not deplacements:
        raise ValueError("Marqueurs du ballon absents sur toute la capture")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        deplacement = np.nanmedian(np.stack(deplacements), axis=0)
    parti = np.where(deplacement > SEUIL_BALLON_MM)[0]
    if len(parti) == 0:
        raise ValueError(f"Le ballon ne bouge pas de plus de {SEUIL_BALLON_MM} mm (frappe non détectée)")
    return int(parti[0])


def detecter_impact(pied, ballons, i_parti, rate):
    """Image de distance pied-ballon minimale, dans les 0,1 s qui précèdent le départ du ballon."""
    n = min(pied.shape[1], ballons.shape[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distance = np.nanmin(np.linalg.norm(ballons[:, :, :n] - pied[None, :, :n], axis=1), axis=0)
    debut = max(0, i_parti - int(round(0.1 * rate)))
    fenetre = distance[debut: i_parti + 1]
    if np.all(np.isnan(fenetre)):
        raise ValueError("Distance pied-ballon inconnue autour de la frappe")
    return debut + int(np.nanargmin(fenetre))


def vitesse_ballon(ballons, i_parti, rate):
    """Vitesse initiale du ballon : pente de la position sur DUREE_VITESSE_BALLON_S après
    son départ (robuste au bruit). Retourne (vecteur en m/s, vitesse en m/s)."""
    nb = max(4, int(round(DUREE_VITESSE_BALLON_S * rate)))
    t = np.arange(nb) / rate
    pentes = []
    for b in ballons:
        seg = b[:, i_parti: i_parti + nb]
        ok = ~np.isnan(seg).any(axis=0)
        if ok.sum() >= 3:
            tt = t[: seg.shape[1]][ok]
            pentes.append([np.polyfit(tt, seg[k, ok], 1)[0] for k in range(3)])
    if not pentes:
        return None, np.nan
    v = np.median(np.array(pentes), axis=0) / 1000
    return v, float(np.linalg.norm(v))


# %%
# ## Inclinaison du tronc (même définition que le script précédent)
#
# Vecteur centre des hanches -> centre des épaules, angle avec la verticale
# dans le plan avant/haut. + = tronc penché vers l'avant (direction de la frappe).

def axe_vertical(ref):
    hanches = milieu(position(ref, "hanche gauche"), position(ref, "hanche droite"))
    epaules = milieu(position(ref, "epaule gauche"), position(ref, "epaule droite"))
    tronc = np.array([moyenne((epaules - hanches)[k]) for k in range(3)])
    k = int(np.argmax(np.abs(tronc))) if AXE_VERTICAL is None else AXE_VERTICAL
    up = np.zeros(3)
    up[k] = np.sign(tronc[k])
    return up


def direction_avant(up, v_ballon, droite_bassin):
    if v_ballon is not None:
        h = v_ballon - (v_ballon @ up) * up
        if np.linalg.norm(h) > 1e-6:
            return h / np.linalg.norm(h), "départ du ballon"
    d = droite_bassin - (droite_bassin @ up) * up
    avant = np.cross(up, d)
    return avant / np.linalg.norm(avant), "ligne des hanches (repli)"


def inclinaison_tronc(essai, up, avant):
    hanches = milieu(position(essai, "hanche gauche"), position(essai, "hanche droite"))
    epaules = milieu(position(essai, "epaule gauche"), position(essai, "epaule droite"))
    return np.degrees(np.arctan2(avant @ (epaules - hanches), up @ (epaules - hanches)))


# %%
# ## Angle du genou droit dans des repères locaux (comme le TD / le 1er script)
#
# - cuisse : origine au centre du genou, z = genou intérieur - genou extérieur,
#   x = (cuisse - origine) x z
# - jambe  : origine à la cheville, z = même axe du genou (pas de paire de
#   marqueurs à la cheville), x = (mollet - origine) x z
# - R_rel = R_cuisse^T . R_jambe, séquence de Cardan ZXY, puis soustraction de
#   l'angle de référence mesuré dans ANAT_REF.

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


NOMS_GENOU = ["cuisse droite", "genou exterieur droit", "genou interieur droite",
              "mollet droite", "cheville droite"]


def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [position(essai, nom) for nom in NOMS_GENOU]
    angles = np.full((3, cuisse.shape[1]), np.nan)
    for i in range(cuisse.shape[1]):
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
    ref = charger_c3d(fichier_ref, "fichier de référence ANAT_REF.c3d")
    angles = angles_genou_droit(ref)
    vitesse = np.convolve(np.linalg.norm(np.diff(position(ref, "cheville droite"), axis=1), axis=0),
                          np.ones(50) / 50, mode="same")
    return angles[:, np.nanargmin(vitesse)]


# %%
# ## Traitement des captures

def traiter_captures():
    ref = charger_c3d(FICHIER_REF, "fichier de référence ANAT_REF.c3d")
    up = axe_vertical(ref)
    angles_ref = angle_reference(FICHIER_REF)
    print("Axe vertical détecté (x, y, z) :", up)
    print("Angle de référence du genou (deg) :", np.round(angles_ref, 1), "\n")

    captures = []
    for mouvement, paires in CAPTURES.items():
        for f_corps, f_ballon in paires:
            print(f"{mouvement} - {f_corps}")
            chemin_corps = DOSSIER / f_corps
            chemin_ballon = DOSSIER / f_ballon

            corps = charger_c3d(chemin_corps, f"capture corps {f_corps}")
            balle = charger_c3d(chemin_ballon, f"capture ballon {f_ballon}")
            rate, n = float(corps.attrs["rate"]), corps.shape[2]
            if abs(float(balle.attrs["rate"]) - rate) > 1e-6 or balle.shape[2] != n:
                print(f"  ATTENTION : fichiers corps/ballon différents ({n} images à {rate:g} Hz "
                      f"contre {balle.shape[2]} à {float(balle.attrs['rate']):g} Hz) : "
                      "synchronisation non garantie")

            ballons = positions_ballon(balle)
            i_parti = depart_ballon(ballons)
            pied = position(corps, MARQUEUR_PIED)
            i_imp = detecter_impact(pied, ballons, i_parti, rate)
            v_vec, v_ballon = vitesse_ballon(ballons, i_parti, rate)

            droite = position(corps, "hanche droite")[:, i_imp] - position(corps, "hanche gauche")[:, i_imp]
            avant, source = direction_avant(up, v_vec, droite)
            tronc = inclinaison_tronc(corps, up, avant)

            vit_pied = vitesse_marqueur(pied, rate)
            flexion = (angles_genou_droit(corps) - angles_ref[:, None])[0]
            omega = derivee(flexion, rate, angle=True)                # deg/s

            nb_i = int(round(FENETRE_IMPACT_S * rate))
            nb_p = int(round(FENETRE_VITESSE_PIED_S * rate))
            nb_o = int(round(FENETRE_OMEGA_S * rate))
            seg_o = omega[max(0, i_imp - nb_o): i_imp + 1]
            if np.all(np.isnan(seg_o)):
                omega_pic, t_pic = np.nan, np.nan
            else:
                k = int(np.nanargmax(np.abs(np.where(np.isnan(seg_o), 0, seg_o))))
                omega_pic = seg_o[k]
                t_pic = (max(0, i_imp - nb_o) + k - i_imp) / rate

            if not np.isnan(t_pic) and (-t_pic >= FENETRE_OMEGA_S - 1.5 / rate or -t_pic <= 1.5 / rate):
                print(f"  ATTENTION : le pic de vitesse du genou ({-t_pic:.2f} s avant l'impact) est au bord "
                      f"de la fenêtre de recherche (0 à {FENETRE_OMEGA_S:g} s) : le vrai pic est peut-être en dehors")

            v_pied = float(np.nanmax(vit_pied[max(0, i_imp - nb_p): i_imp + 1])) \
                if not np.all(np.isnan(vit_pied[max(0, i_imp - nb_p): i_imp + 1])) else np.nan

            m = {"Vitesse pied (m/s)": v_pied,
                 "Vitesse ballon (m/s)": v_ballon,
                 "Rapport ballon/pied": v_ballon / v_pied,
                 "Énergie ballon (J)": 0.5 * MASSE_BALLON_KG * v_ballon ** 2,
                 "Vitesse angulaire pic genou (°/s)": abs(omega_pic),
                 "Angle genou à l'impact (°)": moyenne(flexion[max(0, i_imp - nb_i): i_imp + 1]),
                 "Instant du pic genou (s avant impact)": -t_pic}
            if MASSE_EFFECTIVE_PIED_KG:
                m["Rendement (%)"] = 100 * m["Énergie ballon (J)"] / (0.5 * MASSE_EFFECTIVE_PIED_KG * v_pied ** 2)

            # contrôle de synchronisation : le pic de vitesse du pied doit tomber près de l'impact
            autour = slice(max(0, i_imp - int(0.15 * rate)), i_imp + int(0.15 * rate) + 1)
            ecart = (int(np.nanargmax(vit_pied[autour])) + autour.start - i_imp) / rate \
                if not np.all(np.isnan(vit_pied[autour])) else np.nan

            captures.append(dict(mouvement=mouvement, fichier=f_corps, rate=rate, n=n, i_imp=i_imp,
                                 tronc_brut=tronc, vit_pied=vit_pied, omega=omega, metriques=m,
                                 tronc_impact=moyenne(tronc[max(0, i_imp - nb_i): i_imp + 1])))
            print(f"  impact à {i_imp / rate:.2f} s | avant : {source} | pic de vitesse du pied à "
                  f"{ecart:+.2f} s de l'impact | pied {v_pied:.1f} m/s, ballon {v_ballon:.1f} m/s")

    # grille de temps commune autour de l'impact
    avant_s = min(FENETRE_AVANT_S, min(c["i_imp"] / c["rate"] for c in captures))
    apres_s = min(FENETRE_APRES_S, min((c["n"] - 1 - c["i_imp"]) / c["rate"] for c in captures))
    rate = max(c["rate"] for c in captures)
    grille = np.arange(-int(avant_s * rate), int(apres_s * rate) + 1) / rate
    for c in captures:
        t_rel = (np.arange(c["n"]) - c["i_imp"]) / c["rate"]
        c["g_pied"] = sur_grille(t_rel, c["vit_pied"], grille)
        c["g_omega"] = sur_grille(t_rel, c["omega"], grille)

    print("\nInclinaison du tronc à l'impact (+ = vers l'avant) :")
    for c in captures:
        print(f"  {c['mouvement']:8s} {c['fichier']:26s} {c['tronc_impact']:+6.1f}°")
    moy = [np.mean([c["tronc_impact"] for c in captures if c["mouvement"] == mv]) for mv in CAPTURES]
    if not (moy[0] < moy[1] < moy[2]):
        print("  ATTENTION : l'inclinaison moyenne ne suit pas l'ordre arrière < neutre < avant : "
              "vérifiez la direction 'avant' et l'axe vertical.")
    return grille, captures


# %%
# ## Analyses : lien avec l'inclinaison du tronc (6 captures)
#
# rho de Spearman (lien monotone, robuste), pente (variation de la variable par
# degré de tronc) et effet = pente x amplitude d'inclinaison observée.

def tableau_captures(captures):
    lignes = []
    for c in captures:
        lignes.append({"mouvement": c["mouvement"], "fichier": c["fichier"],
                       "Inclinaison tronc (°)": c["tronc_impact"], **c["metriques"]})
    return pd.DataFrame(lignes).set_index("fichier")


def correlations_tronc(captures):
    x = np.array([c["tronc_impact"] for c in captures])
    lignes = []
    for nom in captures[0]["metriques"]:
        y = np.array([c["metriques"][nom] for c in captures])
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 4 or np.std(y[ok]) == 0:
            lignes.append(dict(variable=nom, rho_spearman=np.nan, r_pearson=np.nan, pente=np.nan,
                               effet_sur_plage=np.nan, variation_relative_pct=np.nan, n=int(ok.sum())))
            continue
        xs, ys = x[ok], y[ok]
        pente = np.polyfit(xs, ys, 1)[0]
        lignes.append(dict(variable=nom,
                           rho_spearman=np.corrcoef(rankdata(xs), rankdata(ys))[0, 1],
                           r_pearson=np.corrcoef(xs, ys)[0, 1],
                           pente=pente, effet_sur_plage=pente * (xs.max() - xs.min()),
                           variation_relative_pct=100 * pente * (xs.max() - xs.min()) / abs(ys.mean()),
                           n=int(ok.sum())))
    return pd.DataFrame(lignes).set_index("variable")


# Grandeurs liées par une simple formule : leur corrélation est automatique, on ne les compare pas.
DERIVEES = {"Rapport ballon/pied": {"Vitesse ballon (m/s)", "Vitesse pied (m/s)", "Énergie ballon (J)", "Rendement (%)"},
            "Rendement (%)": {"Énergie ballon (J)", "Vitesse pied (m/s)", "Rapport ballon/pied", "Vitesse ballon (m/s)"},
            "Énergie ballon (J)": {"Vitesse ballon (m/s)"}}


def _liees(a, b):
    return b in DERIVEES.get(a, ()) or a in DERIVEES.get(b, ())


def correlations_entre_metriques(captures, nb_max=6):
    """Paires de grandeurs (hors tronc) dont la corrélation est significative (p < 0,05) :
    liste de (a, b, rho, n), les plus fortes d'abord."""
    noms = list(captures[0]["metriques"])
    paires = []
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            if _liees(a, b):
                continue
            ya = np.array([c["metriques"][a] for c in captures])
            yb = np.array([c["metriques"][b] for c in captures])
            ok = ~np.isnan(ya) & ~np.isnan(yb)
            if ok.sum() < 4 or np.std(ya[ok]) == 0 or np.std(yb[ok]) == 0:
                continue
            rho = np.corrcoef(rankdata(ya[ok]), rankdata(yb[ok]))[0, 1]
            if abs(rho) >= seuil_spearman(int(ok.sum())) - 1e-9:
                paires.append((abs(rho), a, b, rho, int(ok.sum())))
    paires.sort(reverse=True)
    return [(a, b, rho, n) for _, a, b, rho, n in paires[:nb_max]]


def matrice_chaine(tableau):
    """Corrélations de Spearman entre tronc, genou, pied et ballon (la 'chaîne' de transmission)."""
    cols = ["Inclinaison tronc (°)", "Vitesse angulaire pic genou (°/s)", "Vitesse pied (m/s)",
            "Vitesse ballon (m/s)", "Rapport ballon/pied"]
    return tableau[cols].rank().corr()


# %%
# ## Figures

def _enregistrer(fig, fichier):
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150, facecolor="white", transparent=False)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


def moyenne_courbes(courbes):
    """Moyenne image par image de plusieurs courbes (les NaN sont ignorés)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(courbes), axis=0)


def tracer_profils(grille, captures, fichier=None, afficher_essais=False):
    """Figure 1 : vitesse du pied et vitesse angulaire du genou autour de l'impact.
    Une seule courbe par mouvement = moyenne de ses captures.
    afficher_essais=True ajoute les captures individuelles en traits fins."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor="white")
    for mouvement, couleur in COULEURS.items():
        essais = [c for c in captures if c["mouvement"] == mouvement]
        if not essais:
            continue
        tronc = np.mean([c["tronc_impact"] for c in essais])
        if afficher_essais:
            for c in essais:
                ax1.plot(grille, c["g_pied"], color=couleur, alpha=0.25, linewidth=1)
                ax2.plot(grille, c["g_omega"], color=couleur, alpha=0.25, linewidth=1)
        ax1.plot(grille, moyenne_courbes([c["g_pied"] for c in essais]), color=couleur, linewidth=2.5,
                 label=f"{mouvement} (moyenne de {len(essais)}, tronc {tronc:+.0f}°)")
        ax2.plot(grille, moyenne_courbes([c["g_omega"] for c in essais]), color=couleur, linewidth=2.5)
    for ax in (ax1, ax2):
        ax.axvline(0, color="black", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Vitesse du pied (m/s)")
    ax2.set_ylabel("Vitesse angulaire du genou (°/s)")
    ax2.set_xlabel("Temps par rapport à l'impact (s)")
    ax1.set_title("Pied et genou droit autour de la frappe (moyenne par mouvement)")
    ax1.legend(fontsize=8)
    _enregistrer(fig, fichier)


def tracer_transfert(captures, fichier=None):
    """Figure 2 : vitesse du ballon en fonction de celle du pied, et rapport en fonction du tronc."""
    vp = np.array([c["metriques"]["Vitesse pied (m/s)"] for c in captures])
    vb = np.array([c["metriques"]["Vitesse ballon (m/s)"] for c in captures])
    tr = np.array([c["tronc_impact"] for c in captures])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")

    xx = np.array([np.nanmin(vp) * 0.9, np.nanmax(vp) * 1.1])
    for rapport, style in ((1.0, ":"), (1.5, "--")):
        ax1.plot(xx, rapport * xx, color="grey", linestyle=style, linewidth=1)
        ax1.text(xx[1], rapport * xx[1], f" x{rapport:g}", fontsize=8, color="grey", va="center")
    for c, a, b, t in zip(captures, vp, vb, tr):
        ax1.scatter(a, b, color=COULEURS[c["mouvement"]], s=60)
        ax1.annotate(f"{t:+.0f}°", (a, b), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax1.set_xlabel("Vitesse du pied avant l'impact (m/s)")
    ax1.set_ylabel("Vitesse initiale du ballon (m/s)")
    ax1.set_title("Transfert pied -> ballon\n(étiquette = inclinaison du tronc)", fontsize=10)

    rap = vb / vp
    for c, t, r in zip(captures, tr, rap):
        ax2.scatter(t, r, color=COULEURS[c["mouvement"]], s=60)
    ok = ~np.isnan(tr) & ~np.isnan(rap)
    if ok.sum() >= 3:
        a, b = np.polyfit(tr[ok], rap[ok], 1)
        xs = np.array([tr[ok].min(), tr[ok].max()])
        ax2.plot(xs, a * xs + b, color="grey", linestyle="--", linewidth=1)
    for m, couleur in COULEURS.items():
        ax2.scatter([], [], color=couleur, label=m)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Inclinaison du tronc à l'impact (°, + = vers l'avant)")
    ax2.set_ylabel("Vitesse ballon / vitesse pied")
    ax2.set_title("Efficacité du transfert selon l'inclinaison", fontsize=10)
    _enregistrer(fig, fichier)


PRECISIONS_AXE = {
    "Instant du pic genou (s avant impact)": "secondes avant l'impact\n(0 = pic au moment de l'impact)",
    "Vitesse angulaire pic genou (°/s)": "valeur absolue du pic (°/s)",
}


def _nuage(ax, x, y, captures, titre, xlabel, ylabel, fort=False, incomplet=False):
    for xi, yi, c in zip(x, y, captures):
        ax.scatter(xi, yi, color=COULEURS[c["mouvement"]], s=45, zorder=3)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() >= 3 and np.std(x[ok]) > 0:
        a, b = np.polyfit(x[ok], y[ok], 1)
        xx = np.array([x[ok].min(), x[ok].max()])
        ax.plot(xx, a * xx + b, color="grey", linestyle="--", linewidth=1)
    ax.set_title(titre, fontsize=9, color="tab:red" if incomplet else "black",
                 fontweight="bold" if fort else "normal")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=8)
    if fort:                                       # cadre orange = forte corrélation
        for sp in ax.spines.values():
            sp.set_edgecolor("darkorange")
            sp.set_linewidth(3)


def tracer_nuages(captures, tableau_cor, fichier=None):
    """Figure 3 : chaque grandeur en fonction de l'inclinaison du tronc, classées de la plus à la
    moins corrélée (cadre orange = forte corrélation). En dessous : les autres fortes
    corrélations entre grandeurs (hors tronc)."""
    noms = list(tableau_cor.sort_values("rho_spearman", key=lambda s: s.abs().fillna(-1),
                                        ascending=False).index)
    x = np.array([c["tronc_impact"] for c in captures])
    paires = correlations_entre_metriques(captures)
    nb_col = 3
    l_tronc = int(np.ceil(len(noms) / nb_col))
    l_paires = int(np.ceil(len(paires) / nb_col)) if paires else 0
    hauteurs = [1] * l_tronc + ([0.2] + [1] * l_paires if paires else [])
    fig = plt.figure(figsize=(4.6 * nb_col, 3.4 * (l_tronc + l_paires) + 1), facecolor="white")
    gs = fig.add_gridspec(len(hauteurs), nb_col, height_ratios=hauteurs, hspace=0.75, wspace=0.35)
    marge = 0.05 * (np.nanmax(x) - np.nanmin(x))

    premier = None
    for i, nom in enumerate(noms):
        ax = fig.add_subplot(gs[i // nb_col, i % nb_col])
        premier = premier or ax
        y = np.array([c["metriques"][nom] for c in captures])
        rho, n_ok = tableau_cor.loc[nom, "rho_spearman"], int(tableau_cor.loc[nom, "n"])
        incomplet = n_ok < len(captures)
        titre = nom + ("\ndonnées insuffisantes" if np.isnan(rho) else f"\nρ = {rho:+.2f}")
        if incomplet:
            titre += f"  (seulement {n_ok}/{len(captures)} captures)"
        _nuage(ax, x, y, captures, titre, "Inclinaison du tronc (°)", PRECISIONS_AXE.get(nom, ""),
               fort=(not np.isnan(rho) and abs(rho) >= seuil_spearman(n_ok) - 1e-9), incomplet=incomplet)
        ax.set_xlim(np.nanmin(x) - marge, np.nanmax(x) + marge)
    for m, couleur in COULEURS.items():
        premier.scatter([], [], color=couleur, label=m)
    premier.legend(fontsize=7)

    if paires:
        ax_t = fig.add_subplot(gs[l_tronc, :])
        ax_t.axis("off")
        ax_t.text(0.5, 0.2, "Autres corrélations fortes entre grandeurs (hors tronc)",
                  ha="center", fontsize=11, fontweight="bold")
        for i, (a, b, rho, n_ok) in enumerate(paires):
            ax = fig.add_subplot(gs[l_tronc + 1 + i // nb_col, i % nb_col])
            ya = np.array([c["metriques"][a] for c in captures])
            yb = np.array([c["metriques"][b] for c in captures])
            _nuage(ax, ya, yb, captures, f"{b}\nen fonction de {a}\nρ = {rho:+.2f}", a, b, fort=True)
    fig.suptitle("Grandeurs mesurées à l'impact (cadre orange : corrélation significative, p < 0,05, "
                 "soit |ρ| ≥ 0,89 avec 6 captures)", fontsize=11)
    _enregistrer(fig, fichier)


def tracer_matrice(matrice, fichier=None):
    """Figure 4 : corrélations de Spearman le long de la chaîne tronc -> genou -> pied -> ballon."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="white")
    im = ax.imshow(matrice.values, cmap="coolwarm", vmin=-1, vmax=1)
    etiquettes = [n.replace(" (", "\n(") for n in matrice.columns]
    ax.set_xticks(range(len(matrice)))
    ax.set_yticks(range(len(matrice)))
    ax.set_xticklabels(etiquettes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(etiquettes, fontsize=8)
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            ax.text(j, i, f"{matrice.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Corrélations de Spearman (6 captures)")
    _enregistrer(fig, fichier)


# %%
if __name__ == "__main__":
    print("Dossier du script :", BASE_DIR)
    print("Dossier des fichiers C3D :", DOSSIER)
    print("Fichier de référence :", FICHIER_REF)
    print("Fichier de référence présent :", FICHIER_REF.is_file(), "\n")

    grille, captures = traiter_captures()

    tableau = tableau_captures(captures)
    print("\nRésultats par capture :")
    print(tableau.round(2).to_string())
    tableau.round(3).to_csv("resultats_par_capture.csv")

    cor = correlations_tronc(captures).sort_values("rho_spearman", key=np.abs, ascending=False)
    print("\nLien avec l'inclinaison du tronc (classé par |rho|) :")
    print(cor.round(2).to_string())
    print(f"\nAvec seulement {len(captures)} captures, il faut |r| > {R_CRITIQUE} pour un lien "
          "statistiquement significatif ; en dessous, lisez les résultats comme des tendances.")
    cor.round(3).to_csv("correlations_tronc.csv")
    paires = correlations_entre_metriques(captures)
    if paires:
        print("\nAutres corrélations fortes entre grandeurs (hors tronc) :")
        for a, b, rho, n in paires:
            print(f"  {a}  <->  {b} : rho = {rho:+.2f} (n = {n})")

    tracer_profils(grille, captures, "fig1_profils_pied_genou.png")
    tracer_transfert(captures, "fig2_transfert_pied_ballon.png")
    tracer_nuages(captures, cor, "fig3_grandeurs_vs_tronc.png")
    tracer_matrice(matrice_chaine(tableau), "fig4_chaine_transmission.png")