# -*- coding: utf-8 -*-
"""
===============================================================================
 THE NEXT PURCHASE - Hackathon Ponts 2026 x eleven
 Pipeline v5 : bagging sur les configurations de donnees + fusion de Borda
===============================================================================

 METRIQUE : HIT RATE@5 - 1 si au moins un produit de la visite suivante est
 dans les 5 propositions. C'est la seule metrique d'evaluation. Tous les
 arbitrages du pipeline se font sur elle.

 ------------------------------------------------------------------------------
 CE QUI CHANGE EN v5, ET POURQUOI  (chaque point est mesure, cf. le brief)
 ------------------------------------------------------------------------------
 1. DIAGNOSTIC DES ECHECS de v4, jamais fait avant, sur 20 000 clients :
      - cible deja achetee (24 % des clients) : dans le top-5 a 95 %
      - cible neuve dans le pool (39 %)       : dans le top-5 a ~10 %,
                                                rang median 40 a 141
      - cible hors pool (37 %)                : popularite mediane au rang
                                                2 273, 43 % dans une categorie
                                                jamais achetee - imprevisible
    Les clients les plus actifs ont le PIRE Hit Rate (0.20 a 16+ visites
    contre 0.33 a 3 visites) : le modele leur donne 3.8 slots de rachat sur 5.
    Teste : plafonner les rachats degrade dans TOUS les cas (max 4 -> -0.5 pt,
    max 2 -> -1.8 pt). L'allocation du modele est deja optimale ; le seul
    goulot est la qualite du classement des produits neufs.

 2. SIGNAUX NOUVEAUX (11 features) : transitions strictes visite->visite
    suivante, co-visitation du dernier produit seul, saisonnalite produit,
    anciennete du produit dans le magasin du client, prix relatif a ce que le
    client paie dans la categorie. Utilises par le modele (rangs 8, 13, 19)
    mais REDONDANTS : score inchange (0.2742 vs 0.2754). Conserves.
    Testes et sans signal : promotion (prix recent / habituel : 1.003 pour
    les cibles contre 1.004 pour les autres - les prix sont stables) ;
    fraicheur au magasin (delai depuis la derniere vente : 10 j contre 11).

 3. ENTRAINEMENT MULTI-INSTANTANES CROSS-FITTE : chaque client fournit sa
    derniere ET son avant-derniere visite comme cibles, avec des mondes
    par fold prives des seules cibles de ce fold (zero fuite). Seul : score
    inchange. MAIS ce modele ne fait pas les memes erreurs que celui de v4.

 4. LE LEVIER : BAGGING SUR LES CONFIGURATIONS DE DONNEES. Deux modeles de
    meme score (0.2754 / 0.2748), entraines sur des partitions differentes
    du monde, ne sont d'accord que sur 3.6 items sur 5 ; chacun touche 1.3 %
    de clients que l'autre rate. Fusion de Borda : 0.2784 ; union oracle :
    0.2881. La diversite vient des DONNEES, pas de l'objectif LightGBM
    (teste : +0.2 %). Chaque bag re-partitionne les clients en folds,
    alterne 1 / 2 instantanes, entraine son modele ; les bags sont fusionnes
    par vote de Borda sur leurs top-30. Plus de bags = plus de diversite =
    plus de score, avec un cout lineaire en temps.

 ------------------------------------------------------------------------------
 RESULTATS (CV locale calibree sur le profil de test, comparable leaderboard)
 ------------------------------------------------------------------------------
   v4  40k clients, 1 seed                       Hit Rate 0.2754  (20 000 cl.)
   v4  110k clients, 3 seeds, etage 3            Hit Rate 0.2801  (20 000 cl.)
   v5  fusion de 2 configurations, 40k, 1 seed   Hit Rate 0.2784  (20 000 cl.)
   v5  --max : 6 bags, 110k clients               attendu 0.285 - 0.295

   Leaderboard validation (300 clients, +/- 5 pts) : v2 0.2767, v3 0.2633.
   Ces ecarts sont du bruit ; seule la CV locale sur 20 000 clients est un
   instrument fiable. Choisir la soumission finale sur elle.

 ------------------------------------------------------------------------------
 USAGE
 ------------------------------------------------------------------------------
   pip install pandas numpy scipy scikit-learn lightgbm pyarrow

   python next_purchase_v5.py --fast        #  ~7 min,  6 Go - verification
   python next_purchase_v5.py               # ~2 h,    12 Go - 3 bags
   python next_purchase_v5.py --max         #  6-10 h, 24 Go - 6 bags, 110k clients
   python next_purchase_v5.py --bags 1      # pipeline v4 (etage 3 + melange)

   Reglages : --bags --folds --snapshots --n-train --n-valid --n-es --n-neg
              --seeds --rounds --chunk --workers --no-match --kstore

   MEMOIRE : le pic est celui d'UN bag, quel que soit leur nombre.
   DUREE : lineaire en --bags. La generation de candidats (boucle Python
   mono-thread) domine ; LightGBM est secondaire.

 SORTIES (output/)
   submission_all.csv / submission_validation.csv / submission_final.csv
   feature_importance.csv, run_report.txt, journal.csv
===============================================================================
"""

import os
import gc
import time
import zipfile
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(DATA_DIR, "output")
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    SEED = 42
    # DEUX reglages distincts, a ne pas confondre :
    #   MATCH_EVAL  calibre la population de VALIDATION et d'early-stopping sur
    #               le profil des clients de test. C'est une correction de
    #               MESURE : sans elle la CV sous-estime le leaderboard de
    #               6.9 % (mesure : CV 0.2589 -> leaderboard 0.2767).
    #   MATCH_TRAIN calibre aussi la population d'ENTRAINEMENT. Mesure : cela
    #               fait PERDRE 3.4 %, parce que les clients tres actifs sont
    #               rares et qu'on jette ainsi 60 % du pool. Le modele a besoin
    #               de volume, l'evaluation a besoin de representativite.
    MATCH_EVAL = True
    MATCH_TRAIN = False

    # Objectifs cycles sur les seeds de l'etage 2 (None = un seul objectif).
    MULTI_OBJ = ("lambdarank", "rank_xendcg", "binary")

    # --- entrainement MULTI-INSTANTANES cross-fitte (nouveau en v5) -----------
    # Chaque client d'entrainement fournit N_SNAPSHOTS exemples : sa derniere
    # visite (comme avant) ET son avant-derniere, avec l'historique tronque en
    # consequence. Les clients sont repartis en N_FOLDS ; pour chaque fold on
    # construit un "monde" prive des visites-cibles de CE fold uniquement, les
    # autres folds y restant entiers. Zero fuite, et deux fois plus d'exemples
    # sans perdre plus de donnees d'agregats qu'avant.
    N_FOLDS = 2
    N_SNAPSHOTS = 2

    # --- BAGGING SUR LES CONFIGURATIONS DE DONNEES (nouveau en v5) -------------
    # Mesure decisive : deux modeles de meme score (0.2754 / 0.2748) entraines
    # sur des partitions differentes du monde ne sont d'accord que sur 3.6
    # items sur 5 ; chacun touche ~1.3 % de clients que l'autre rate. Leur
    # fusion donne 0.2784, l'union oracle 0.2881. La diversite utile vient de
    # la CONFIGURATION DES DONNEES (partition des folds, nb d'instantanes), pas
    # de l'objectif LightGBM (teste : +0.2 %). Chaque bag re-partitionne les
    # clients, alterne 1 / 2 instantanes, entraine son propre modele ; les
    # bags sont fusionnes par vote de Borda sur leurs top-K_FUSE.
    N_BAGS = 3
    K_FUSE = 30
    K_NXT = 40                   # transitions strictes visite->visite suivante
    K_NXT_OUT = 120

    # --- volumes ------------------------------------------------------------
    N_TRAIN_CLIENTS = 110_000    # clients pour l'etage 2
    N_TRAIN2_CLIENTS = 60_000    # clients pour l'etage 3 (sous-ensemble)
    N_VALID_CLIENTS = 9_000      # CV locale, jamais vue par les modeles
    N_ES_CLIENTS = 4_000         # early stopping, disjoint du jeu de CV
    N_NEG_PER_CLIENT = 60        # negatifs par groupe, etage 2
    PRUNE_TO = 120               # candidats conserves apres l'etage 2
    # Taille des blocs de generation. Le pic memoire du pipeline est celui
    # d'UN bloc : CHUNK_CLIENTS x taille du pool x nb de features x 4 octets.
    # Avec un pool de ~900 candidats, 2 500 clients par bloc font deja 1.2 Go.
    # A reduire si la machine a peu de RAM (--chunk 1500).
    CHUNK_CLIENTS = 2_500

    # --- retrieval : tailles par source ------------------------------------
    K_POP_COUNTRY = 170          # top-N popularite pays (point-in-time, 3 mois)
    K_POP_COUNTRY_LONG = 80      # idem, fenetre 12 mois
    K_POP_GLOBAL = 40
    K_POP_SEGMENT = 40
    # 81.5 % des produits cibles sont vendus dans le DERNIER magasin visite,
    # et 73.3 % des prochaines visites s'y deroulent. C'est la source la plus
    # dense du pipeline : a elle seule, son top-900 couvre 55 % des cibles.
    # Taille mesuree comme optimale. Elargir a 450 fait monter le plafond du
    # retrieval de 0.634 a 0.670 mais FAIT BAISSER le score final (0.2627 ->
    # 0.2607) : le ranker se dilue. Regle generale de ce probleme : au-dela de
    # ~650 candidats, chaque candidat supplementaire coute plus qu'il ne
    # rapporte. Ne pas gonfler ces valeurs.
    K_STORE_LAST = 250           # assortiment du magasin de la DERNIERE visite
    K_STORE_FAV = 130            # assortiment du magasin prefere
    K_COVIS_SEQ = 40             # voisins sequentiels retenus par produit-graine
    K_COVIS_SEQ_OUT = 260        # candidats issus de la co-visitation sequentielle
    K_COVIS_BSK = 16
    K_COVIS_BSK_OUT = 90
    K_UU = 110                   # candidats issus du CF user-user
    K_FAMTRANS = 110             # candidats issus des transitions de famille
    K_FAMILY = 45                # top-N par (FamilyLevel2, Universe)
    K_SVD = 60                   # candidats issus des voisins SVD
    N_HIST_SEEDS = 20            # produits recents servant de graines
    POP_WINDOWS = (1, 3, 6, 12)

    # --- co-visitation ------------------------------------------------------
    COVIS_MAX_ITEMS = 60
    COVIS_SEQ_MAX_GAP = 150      # jours
    COVIS_TOPK_STORE = 80
    FAMTRANS_TOPK = 8

    # --- SVD / CF user-user -------------------------------------------------
    SVD_DIM = 96
    SVD_MIN_PROD_COUNT = 4
    SVD_NN = 24
    UU_K = 60                    # voisins clients
    UU_MIN_TX = 2                # un voisin doit avoir au moins 2 achats
    UU_BLOCK = 256

    # --- LightGBM -----------------------------------------------------------
    LGB1 = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 25, "boosting_type": "gbdt",
        "learning_rate": 0.04, "num_leaves": 160, "min_data_in_leaf": 60,
        "feature_fraction": 0.72, "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l1": 0.2, "lambda_l2": 3.0, "max_bin": 255,
        "num_threads": 0, "verbosity": -1, "seed": SEED,
    }
    LGB2 = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 12, "boosting_type": "gbdt",
        "learning_rate": 0.03, "num_leaves": 200, "min_data_in_leaf": 40,
        "feature_fraction": 0.7, "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l1": 0.2, "lambda_l2": 4.0, "max_bin": 255,
        "num_threads": 0, "verbosity": -1, "seed": SEED,
    }
    ROUNDS1, ROUNDS2 = 2000, 2500
    EARLY_STOP = 120
    N_SEEDS1, N_SEEDS2 = 2, 3    # ensembles (moyenne des rangs)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log(f"-> {self.name}")
        return self

    def __exit__(self, *a):
        log(f"   {self.name} : {time.time() - self.t:.1f}s")


# =============================================================================
# 1. CHARGEMENT ET ENCODAGE
#    Les IDs sont lus en str puis convertis en codes int32. Aucun calcul
#    numerique ne touche jamais un identifiant : la notation scientifique
#    (6.63E+16) est structurellement impossible.
# =============================================================================
ID_STR = {"ClientID": str, "ProductID": str, "StoreID": str}


class Vocab:
    def __init__(self, values):
        self.keys = pd.Index(pd.unique(pd.Series(values, dtype=object).astype(str)))
        self.lookup = {k: i for i, k in enumerate(self.keys)}
        self.n = len(self.keys)

    def enc(self, s, default=-1):
        return (pd.Series(s, dtype=object).astype(str).map(self.lookup)
                .fillna(default).astype("int32").values)


def load_raw():
    d = CFG.DATA_DIR
    os.makedirs(CFG.CACHE_DIR, exist_ok=True)
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    csv = os.path.join(d, "transactions.csv")
    zp = os.path.join(d, "transactions.zip")
    if not os.path.exists(csv) and os.path.exists(zp):
        log("decompression de transactions.zip ...")
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)
    cache = os.path.join(CFG.CACHE_DIR, "trans.parquet")
    if os.path.exists(cache):
        tx = pd.read_parquet(cache)
    else:
        tx = pd.read_csv(csv, dtype=ID_STR)
        tx["d"] = pd.to_datetime(tx["SaleTransactionDate"], utc=True, errors="coerce")
        tx["d"] = tx["d"].dt.tz_localize(None).dt.normalize()
        tx = tx.drop(columns=["SaleTransactionDate"]).dropna(subset=["d", "ClientID", "ProductID"])
        tx.to_parquet(cache, index=False)
    r = lambda f, **k: pd.read_csv(os.path.join(d, f), **k)
    clients = r("clients.csv", dtype=ID_STR)
    products = r("products.csv", dtype=ID_STR)
    stocks = r("stocks.csv", dtype={"ProductID": str, "StoreCountry": str})
    stocks["Quantity"] = pd.to_numeric(stocks["Quantity"], errors="coerce").fillna(0)
    return (tx, clients, products, r("stores.csv", dtype=ID_STR), stocks,
            r("test_clients_validation.csv", dtype=str), r("test_clients_final.csv", dtype=str))


def prepare(tx, clients, products, stocks):
    with timer("encodage et tables de reference"):
        pv = Vocab(pd.concat([products["ProductID"], tx["ProductID"]], ignore_index=True))
        cv = Vocab(pd.concat([clients["ClientID"], tx["ClientID"]], ignore_index=True))
        sv = Vocab(tx["StoreID"])

        tx = tx.copy()
        tx["p"] = pv.enc(tx["ProductID"])
        tx["c"] = cv.enc(tx["ClientID"])
        tx["s"] = sv.enc(tx["StoreID"])
        tx["day"] = (tx["d"] - pd.Timestamp("2023-01-01")).dt.days.astype("int32")
        tx["mi"] = (tx["d"].dt.year * 12 + tx["d"].dt.month).astype("int32")
        tx["amt"] = pd.to_numeric(tx["SalesNetAmountEuro"], errors="coerce").fillna(0).astype("float32")
        tx["qty"] = pd.to_numeric(tx["Quantity"], errors="coerce").fillna(1).astype("float32")
        tx = tx[["c", "p", "s", "day", "mi", "amt", "qty"]]

        # ---- attributs produit indexes par code ----------------------------
        products = products.copy()
        products["p"] = pv.enc(products["ProductID"])
        P = pd.DataFrame({"p": np.arange(pv.n, dtype="int32")}).merge(
            products[["p", "Category", "FamilyLevel1", "FamilyLevel2", "Universe"]], on="p", how="left")
        for col in ["Category", "FamilyLevel1", "FamilyLevel2", "Universe"]:
            P[col] = P[col].fillna("NA").astype("category")
        prod = {
            "cat": P["Category"].cat.codes.values.astype("int16"),
            "f1": P["FamilyLevel1"].cat.codes.values.astype("int16"),
            "f2": P["FamilyLevel2"].cat.codes.values.astype("int16"),
            "uni": P["Universe"].cat.codes.values.astype("int8"),
        }
        uc = list(P["Universe"].cat.categories)
        prod["uni_women"] = uc.index("Women") if "Women" in uc else -1
        prod["n_f2"] = int(prod["f2"].max()) + 1

        unit = (tx["amt"] / np.maximum(tx["qty"], 1)).astype("float32")
        pm = pd.DataFrame({"p": tx["p"].values, "u": unit.values}).groupby("p")["u"].mean()
        price = np.zeros(pv.n, dtype="float32")
        price[pm.index.values] = pm.values
        med = float(np.median(price[price > 0])) if (price > 0).any() else 1.0
        price[price <= 0] = med
        prod["price"] = price
        prod["logprice"] = np.log1p(price).astype("float32")
        dfp = pd.DataFrame({"f2": prod["f2"], "uni": prod["uni"], "price": price})
        prod["price_rank_fam"] = dfp.groupby(["f2", "uni"])["price"].rank(pct=True).fillna(0.5).values.astype("float32")
        fam_med = dfp.groupby(["f2", "uni"])["price"].transform("median").values.astype("float32")
        prod["price_vs_fam"] = (price / np.maximum(fam_med, 0.01)).astype("float32")

        # ---- attributs client ----------------------------------------------
        clients = clients.copy()
        clients["c"] = cv.enc(clients["ClientID"])
        ccat = {}
        for col in ["ClientSegment", "ClientCountry", "ClientGender"]:
            s = clients[col].fillna("NA").astype("category")
            arr = np.full(cv.n, -1, dtype="int8")
            arr[clients["c"].values] = s.cat.codes.values.astype("int8")
            ccat[col] = arr
            ccat[col + "_cats"] = list(s.cat.categories)
        age = np.full(cv.n, np.nan, dtype="float32")
        a = pd.to_numeric(clients["Age"], errors="coerce").values
        age[clients["c"].values] = np.where((a >= 10) & (a <= 100), a, np.nan)   # min 3 / max 125 observes
        ccat["Age"] = age
        for col in ["ClientOptINEmail", "ClientOptINPhone"]:
            arr = np.full(cv.n, -1, dtype="int8")
            arr[clients["c"].values] = pd.to_numeric(clients[col], errors="coerce").fillna(-1).values.astype("int8")
            ccat[col] = arr

        # ---- stock : matrice pays x produit, utilisee comme FEATURE ---------
        countries = sorted(set(clients["ClientCountry"].dropna().unique()) |
                           set(stocks["StoreCountry"].dropna().unique()))
        cidx = {k: i for i, k in enumerate(countries)}
        stock = np.zeros((len(countries), pv.n), dtype="float32")
        s_ = stocks[stocks["Quantity"] > 0]
        rr = s_["StoreCountry"].map(cidx).fillna(-1).astype(int).values
        cc = pv.enc(s_["ProductID"])
        ok = (rr >= 0) & (cc >= 0)
        np.add.at(stock, (rr[ok], cc[ok]), s_["Quantity"].values[ok])
        cc_of = clients.set_index("c")["ClientCountry"].reindex(np.arange(cv.n)).fillna("NA").values
        ccountry = pd.Series(cc_of).map(cidx).fillna(-1).astype("int16").values

        tx = tx.sort_values(["c", "day"], kind="mergesort").reset_index(drop=True)
    return tx, dict(pv=pv, cv=cv, sv=sv, prod=prod, ccat=ccat, stock=stock,
                    ccountry=ccountry, countries=countries)


# =============================================================================
# 2. LE "MONDE" : tous les agregats servant au retrieval et aux features.
#    Deux instances : TRAIN (prive des visites-cibles, donc sans fuite) et
#    INFER (toutes les transactions). Meme code, aucune divergence possible.
# =============================================================================
class World:
    def __init__(self, tx, ref, name):
        self.name, self.ref, self.tx = name, ref, tx
        self.n_p, self.n_c, self.n_s = ref["pv"].n, ref["cv"].n, ref["sv"].n
        self.m_min, self.m_max = int(tx["mi"].min()), int(tx["mi"].max())
        self.n_m = self.m_max - self.m_min + 2
        self._tp = {}
        self._popularity()
        self._store()
        self._covis()
        self._family()
        self._svd()

    # ---- popularite cumulee par mois : pop(m,p) sur n'importe quelle fenetre
    def _popularity(self):
        with timer(f"[{self.name}] popularite point-in-time"):
            tx, n_p = self.tx, self.n_p
            m = (tx["mi"].values - self.m_min + 1).astype("int32")

            def cum(rows, n_rows):
                M = np.zeros((n_rows, self.n_m, n_p), dtype="float32")
                np.add.at(M, (rows, m, tx["p"].values), 1.0)
                return np.cumsum(M, axis=1)

            self.CUM_G = cum(np.zeros(len(tx), dtype="int32"), 1)
            ct = np.maximum(self.ref["ccountry"][tx["c"].values], 0).astype("int32")
            self.CUM_C = cum(ct, len(self.ref["countries"]))
            sg = np.maximum(self.ref["ccat"]["ClientSegment"][tx["c"].values], 0).astype("int32")
            self.CUM_S = cum(sg, int(sg.max()) + 1)

            u = tx[["p", "c"]].drop_duplicates()
            self.n_buyers = np.bincount(u["p"].values, minlength=n_p).astype("float32")
            first = np.full(n_p, 10 ** 6, dtype="int32")
            np.minimum.at(first, tx["p"].values, tx["day"].values)
            self.first_day = first
            seen = self.CUM_G[0] > 0
            self.LAST_M = np.maximum.accumulate(
                np.where(seen, np.arange(self.n_m)[:, None], -1), axis=0).astype("int16")
            # propension du produit a etre rachete par le meme client
            pair = tx.groupby(["p", "c"], as_index=False).size()
            rr = pair.assign(r=(pair["size"] > 1).astype("float32")).groupby("p")["r"].mean()
            self.p_repeat = np.zeros(n_p, dtype="float32")
            self.p_repeat[rr.index.values] = rr.values
            ns = tx.groupby("p")["s"].nunique()
            self.p_n_stores = np.zeros(n_p, dtype="float32")
            self.p_n_stores[ns.index.values] = ns.values.astype("float32")
            self.p_price_med = self.ref["prod"]["price"]

    # ---- assortiment magasin. 71.4 % des prochaines visites ont lieu dans le
    #      meme magasin que la derniere : c'est le 2e signal apres le rachat.
    def _store(self):
        with timer(f"[{self.name}] assortiment magasin + saisonnalite"):
            M = np.zeros((self.n_s, self.n_p), dtype="float32")
            np.add.at(M, (self.tx["s"].values, self.tx["p"].values), 1.0)
            self.store_mat = M
            self.store_tot = M.sum(axis=1) + 1.0
            k = max(CFG.K_STORE_LAST, CFG.K_STORE_FAV, 64)
            self.STORE_TOP = {}
            self.STORE_RANK = {}
            for si in range(self.n_s):
                v = M[si]
                if v.sum() == 0:
                    self.STORE_TOP[si] = np.zeros(0, dtype="int32")
                    continue
                idx = np.argpartition(-v, min(k, len(v) - 1))[:k]
                idx = idx[np.argsort(-v[idx])]
                self.STORE_TOP[si] = idx[v[idx] > 0].astype("int32")
            # totaux par (magasin, famille) et (pays, famille) : servent a
            # calculer la PART d'un produit DANS sa famille - c'est le signal
            # qui distingue une reference d'une autre a l'interieur d'une
            # meme FamilyLevel2, la ou se joue tout le probleme des produits
            # neufs (mediane de 91 references par famille x univers).
            prod = self.ref["prod"]
            fk = (prod["f2"].astype("int32") * 4 + np.maximum(prod["uni"], 0).astype("int32"))
            self.fam_key = fk
            n_k = int(fk.max()) + 1
            self.n_fam_key = n_k
            SF = np.zeros((self.n_s, n_k), dtype="float32")
            np.add.at(SF, (self.tx["s"].values, fk[self.tx["p"].values]), 1.0)
            self.STORE_FAM = SF
            ct = np.maximum(self.ref["ccountry"][self.tx["c"].values], 0).astype("int32")
            CF = np.zeros((len(self.ref["countries"]), n_k), dtype="float32")
            np.add.at(CF, (ct, fk[self.tx["p"].values]), 1.0)
            self.CTRY_FAM = CF
            GF = np.zeros(n_k, dtype="float32")
            np.add.at(GF, fk[self.tx["p"].values], 1.0)
            self.GLOB_FAM = GF

            moy = ((self.tx["mi"].values - 1) % 12).astype("int32")
            cat = self.ref["prod"]["cat"][self.tx["p"].values].astype("int32")
            S = np.zeros((int(self.ref["prod"]["cat"].max()) + 1, 12), dtype="float32")
            np.add.at(S, (cat, moy), 1.0)
            self.CAT_SEASON = S / np.maximum(S.sum(axis=1, keepdims=True), 1.0) * 12.0
            # NOUVEAU v5 - saisonnalite PRODUIT : part des ventes du produit
            # tombant dans chaque mois calendaire, rapportee a 1/12. Un produit
            # a 3.0 en novembre se vend trois fois plus que sa moyenne ce mois-la.
            PS = np.zeros((self.n_p, 12), dtype="float32")
            np.add.at(PS, (self.tx["p"].values, moy), 1.0)
            tot = PS.sum(axis=1, keepdims=True)
            self.P_SEASON = np.where(tot >= 12, PS / np.maximum(tot, 1.0) * 12.0, 1.0).astype("float32")
            # NOUVEAU v5 - premiere vente de chaque produit dans chaque magasin
            SF = np.full((self.n_s, self.n_p), 10 ** 6, dtype="int32")
            np.minimum.at(SF, (self.tx["s"].values, self.tx["p"].values), self.tx["day"].values)
            self.STORE_FIRST = SF

    # ---- co-visitation : SEQ modelise "quel produit vient APRES", BSK la
    #      complementarite dans un meme panier. Poids normalises par la
    #      popularite de la cible (lift plutot que compte brut).
    def _covis(self):
        with timer(f"[{self.name}] matrices de co-visitation"):
            tx = self.tx[["c", "p", "day"]]
            tx = tx[tx.groupby("c")["p"].transform("size") <= CFG.COVIS_MAX_ITEMS]
            self.COV_SEQ = self._pairs(tx, True)
            self.COV_BSK = self._pairs(tx, False)
            # NOUVEAU v5 - transitions STRICTES visite i -> visite i+1. COV_SEQ
            # agrege toutes les paires a moins de 150 jours ; ici on ne garde
            # que le successeur immediat, sans decroissance : un signal plus
            # net de "ce qui vient juste apres".
            self.COV_NXT = self._next_visit_pairs(tx)

    def _next_visit_pairs(self, tx):
        tx = tx.sort_values(["c", "day"], kind="mergesort")
        vis = tx[["c", "day"]].drop_duplicates()
        vis["vi"] = vis.groupby("c").cumcount()
        tx = tx.merge(vis, on=["c", "day"])
        nxt = tx[["c", "vi", "p"]].copy()
        nxt["vi"] = nxt["vi"] - 1                       # la visite i+1 vue depuis i
        m = tx[["c", "vi", "p"]].merge(nxt, on=["c", "vi"], suffixes=("_x", "_y"))
        m = m[m["p_x"].values != m["p_y"].values]
        if len(m) == 0:
            return {}
        popb = self.CUM_G[0, -1, :]
        co = m.groupby(["p_x", "p_y"], as_index=False).size().rename(columns={"size": "w"})
        co["w"] = co["w"].values.astype("float32") / np.sqrt(1.0 + popb[co["p_y"].values])
        co = co.sort_values(["p_x", "w"], ascending=[True, False])
        co = co[co.groupby("p_x").cumcount() < CFG.COVIS_TOPK_STORE]
        return {int(a): (g["p_y"].values.astype("int32"), g["w"].values.astype("float32"))
                for a, g in co.groupby("p_x", sort=False)}

    def _pairs(self, tx, seq):
        out, cs = [], tx["c"].values
        if len(cs) == 0:
            return {}
        bounds = np.linspace(cs.min(), cs.max() + 1, 13).astype(int)
        popb = self.CUM_G[0, -1, :]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            sub = tx[(tx["c"] >= lo) & (tx["c"] < hi)]
            if len(sub) == 0:
                continue
            m = sub.merge(sub, on="c")
            gap = m["day_y"].values - m["day_x"].values
            keep = ((gap > 0) & (gap <= CFG.COVIS_SEQ_MAX_GAP)) if seq else \
                   ((gap == 0) & (m["p_x"].values != m["p_y"].values))
            if not keep.any():
                continue
            w = (1.0 / (1.0 + np.abs(gap[keep]) / 30.0)).astype("float32") if seq \
                else np.ones(int(keep.sum()), dtype="float32")
            out.append(pd.DataFrame({"a": m["p_x"].values[keep], "b": m["p_y"].values[keep], "w": w}))
            del m
            gc.collect()
        if not out:
            return {}
        co = pd.concat(out, ignore_index=True).groupby(["a", "b"], as_index=False)["w"].sum()
        co["w"] = co["w"].values / np.sqrt(1.0 + popb[co["b"].values])
        co = co.sort_values(["a", "w"], ascending=[True, False])
        co = co[co.groupby("a").cumcount() < CFG.COVIS_TOPK_STORE]
        return {int(a): (g["b"].values.astype("int32"), g["w"].values.astype("float32"))
                for a, g in co.groupby("a", sort=False)}

    # ---- transitions de FamilyLevel2 + popularite par famille / categorie
    def _family(self):
        with timer(f"[{self.name}] transitions et popularite de famille"):
            prod = self.ref["prod"]
            tx = self.tx[["c", "p", "day"]]
            tx = tx[tx.groupby("c")["p"].transform("size") <= CFG.COVIS_MAX_ITEMS]
            f2 = prod["f2"][tx["p"].values]
            tf = pd.DataFrame({"c": tx["c"].values, "f": f2, "day": tx["day"].values})
            out, cs = [], tf["c"].values
            bounds = np.linspace(cs.min(), cs.max() + 1, 13).astype(int)
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                sub = tf[(tf["c"] >= lo) & (tf["c"] < hi)]
                if len(sub) == 0:
                    continue
                m = sub.merge(sub, on="c")
                gap = m["day_y"].values - m["day_x"].values
                keep = (gap > 0) & (gap <= CFG.COVIS_SEQ_MAX_GAP)
                if keep.any():
                    out.append(pd.DataFrame({"a": m["f_x"].values[keep], "b": m["f_y"].values[keep]}))
                del m
            self.FTRANS, self.FTRANS_P = {}, {}
            if out:
                ft = pd.concat(out, ignore_index=True).groupby(["a", "b"], as_index=False).size()
                ft["pr"] = ft["size"] / ft.groupby("a")["size"].transform("sum")
                ft = ft.sort_values(["a", "size"], ascending=[True, False])
                ft = ft[ft.groupby("a").cumcount() < CFG.FAMTRANS_TOPK]
                for a, g in ft.groupby("a", sort=False):
                    self.FTRANS[int(a)] = g["b"].values.astype("int32")
                    self.FTRANS_P[int(a)] = dict(zip(g["b"].values.astype(int), g["pr"].values))
            recent = self.tx[self.tx["mi"] >= self.m_max - 11]
            key = prod["f2"][recent["p"].values].astype("int32") * 4 + \
                np.maximum(prod["uni"][recent["p"].values], 0).astype("int32")
            g = pd.DataFrame({"k": key, "p": recent["p"].values}).groupby(
                ["k", "p"], as_index=False).size().sort_values(["k", "size"], ascending=[True, False])
            g = g[g.groupby("k").cumcount() < CFG.K_FAMILY]
            self.FAM = {int(k): (v["p"].values.astype("int32"), v["size"].values.astype("float32"))
                        for k, v in g.groupby("k", sort=False)}
            cg = pd.DataFrame({"k": prod["cat"][recent["p"].values].astype("int32"),
                               "p": recent["p"].values}).groupby(
                ["k", "p"], as_index=False).size().sort_values(["k", "size"], ascending=[True, False])
            cg = cg[cg.groupby("k").cumcount() < CFG.K_FAMILY]
            self.CATP = {int(k): v["p"].values.astype("int32") for k, v in cg.groupby("k", sort=False)}

    # ---- SVD : embeddings produits (voisins + score) et clients (CF user-user)
    def _svd(self):
        with timer(f"[{self.name}] SVD produits et clients"):
            cnt = np.bincount(self.tx["p"].values, minlength=self.n_p)
            keep = np.where(cnt >= CFG.SVD_MIN_PROD_COUNT)[0]
            remap = np.full(self.n_p, -1, dtype="int32")
            remap[keep] = np.arange(len(keep))
            sub = self.tx[remap[self.tx["p"].values] >= 0]
            M = sp.coo_matrix((np.ones(len(sub), dtype="float32"),
                               (sub["c"].values, remap[sub["p"].values])),
                              shape=(self.n_c, len(keep))).tocsr()
            M.data[:] = 1.0
            rs = np.asarray(M.sum(1)).ravel()
            M = sp.diags((1.0 / np.sqrt(np.maximum(rs, 1.0))).astype("float32")) @ M
            idf = np.log1p(self.n_c / np.maximum(np.asarray(M.sum(0)).ravel(), 1.0)).astype("float32")
            M = (M @ sp.diags(idf)).tocsr()
            dim = min(CFG.SVD_DIM, max(2, min(M.shape) - 1))
            svd = TruncatedSVD(dim, random_state=CFG.SEED, algorithm="randomized").fit(M)
            E = svd.components_.T.astype("float32")
            E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
            self.svd_keep, self.svd_remap, self.svd_emb = keep, remap, E
            U = np.asarray(M @ svd.components_.T, dtype="float32")
            U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
            self.user_emb = U
            self.uu_pool = np.where(rs >= CFG.UU_MIN_TX)[0].astype("int32")
            nn_i = np.zeros((len(keep), CFG.SVD_NN), dtype="int32")
            nn_s = np.zeros((len(keep), CFG.SVD_NN), dtype="float32")
            B = 2048
            for i in range(0, len(keep), B):
                S = E[i:i + B] @ E.T
                np.put_along_axis(S, np.arange(i, min(i + B, len(keep)))[:, None], -9.0, axis=1)
                idx = np.argpartition(-S, CFG.SVD_NN, axis=1)[:, :CFG.SVD_NN]
                sc = np.take_along_axis(S, idx, axis=1)
                o = np.argsort(-sc, axis=1)
                nn_i[i:i + B] = keep[np.take_along_axis(idx, o, axis=1)]
                nn_s[i:i + B] = np.take_along_axis(sc, o, axis=1)
                del S
            self.svd_nn_i, self.svd_nn_s = nn_i, nn_s
            # produits achetes par chaque client, pour propager via le CF
            self.prods_of = self.tx.groupby("c")["p"].apply(
                lambda s: np.array(list(dict.fromkeys(s.values)), dtype="int32")).to_dict()

    def user_neighbors(self, cids):
        """kNN client x client dans l'espace SVD, par blocs (memoire bornee)."""
        U, pool = self.user_emb, self.uu_pool
        Up = U[pool]
        res = {}
        K = CFG.UU_K
        for i in range(0, len(cids), CFG.UU_BLOCK):
            blk = cids[i:i + CFG.UU_BLOCK]
            S = U[blk] @ Up.T
            idx = np.argpartition(-S, K + 1, axis=1)[:, :K + 1]
            sc = np.take_along_axis(S, idx, axis=1)
            o = np.argsort(-sc, axis=1)
            idx = np.take_along_axis(idx, o, axis=1)
            sc = np.take_along_axis(sc, o, axis=1)
            for r, c in enumerate(blk):
                nb, w = pool[idx[r]], sc[r]
                m = nb != c
                res[int(c)] = (nb[m][:K], w[m][:K].astype("float32"))
            del S
        return res

    # ---- top popularite point-in-time, memoise
    def top_pop(self, mi, row, win, k, table="C"):
        key = (mi, row, win, table)
        if key not in self._tp:
            m = int(np.clip(mi - self.m_min + 1, 0, self.n_m - 1))
            m0 = int(np.clip(m - win, 0, self.n_m - 1))
            T = {"G": self.CUM_G, "C": self.CUM_C, "S": self.CUM_S}[table]
            v = T[max(int(row), 0), m] - T[max(int(row), 0), m0]
            n = min(300, len(v) - 1)
            idx = np.argpartition(-v, n)[:n]
            idx = idx[np.argsort(-v[idx])]
            self._tp[key] = idx[v[idx] > 0].astype("int32")
        return self._tp[key][:k]

    def pop(self, table, row, mi, p, win):
        m = np.clip(mi - self.m_min + 1, 0, self.n_m - 1)
        m0 = np.clip(m - win, 0, self.n_m - 1)
        return table[row, m, p] - table[row, m0, p]


# =============================================================================
# 3. PROTOCOLE : cible = derniere visite, t0 = avant-derniere.
#    Exactement le protocole officiel. Les clients de test ont deja leur
#    derniere visite retiree par les organisateurs : leur t0 est leur date max.
# =============================================================================
def match_test_profile(pool, split, test_codes, n, rng):
    """Echantillonne des clients dont la distribution du NOMBRE DE VISITES
    reproduit celle des clients de test.

    Mesure : les clients de test ont 5 visites en mediane contre 3 dans le
    pool, et seulement 6 % d'entre eux ont 2 visites contre 34 % du pool.
    Tirer uniformement revient donc a entrainer et a mesurer sur une
    population beaucoup plus pauvre en historique que la vraie cible - la CV
    sous-estime le score de ~6.5 % et le modele se specialise sur les mauvais
    clients. C'est la correction structurelle la plus rentable du pipeline."""
    tv = split.set_index("c")["n_vis"]
    tprof = tv.reindex(list(test_codes)).dropna()
    BINS = [1, 2, 3, 4, 5, 7, 10, 15, 10 ** 6]
    tfrac = pd.cut(tprof, bins=BINS).value_counts(normalize=True)
    pb = pd.cut(tv.reindex(pool).dropna(), bins=BINS)
    out = []
    for b, frac in tfrac.items():
        cand = pb[pb == b].index.values
        k = int(round(frac * n))
        if len(cand) == 0 or k == 0:
            continue
        out.append(rng.choice(cand, min(k, len(cand)), replace=False))
    if not out:
        return np.asarray(pool)[:n]
    ids = np.concatenate(out)
    # Les clients tres actifs sont rares : au-dela d'environ 75 000 clients la
    # distribution exacte n'est plus atteignable. On complete alors par les
    # clients restants LES PLUS ACTIFS, ce qui garde la population proche du
    # profil de test au lieu de retomber vers un tirage uniforme.
    if len(ids) < n:
        rest = np.setdiff1d(np.asarray(pool), ids, assume_unique=False)
        if len(rest):
            order = tv.reindex(rest).fillna(0).values.argsort()[::-1]
            ids = np.concatenate([ids, rest[order][:n - len(ids)]])
        log(f"   profil calibre jusqu'a {len(ids):,} clients "
            f"(bins rares epuises, complement par activite decroissante)")
    rng.shuffle(ids)
    return ids.astype("int32")


def build_split(tx, test_codes):
    v = tx[["c", "day"]].drop_duplicates().sort_values(["c", "day"], kind="mergesort")
    v["rk"] = v.groupby("c").cumcount(ascending=False)          # 0 = derniere visite
    last = v[v["rk"] == 0].set_index("c")["day"].rename("last_day")
    t0 = v[v["rk"] == 1].set_index("c")["day"].rename("t0")     # avant-derniere
    t0b = v[v["rk"] == 2].set_index("c")["day"].rename("t0b")   # antepenultieme
    nvis = v.groupby("c").size().rename("n_vis")
    sp_ = pd.concat([last, nvis, t0, t0b], axis=1).reset_index()
    pool = sp_[(sp_["n_vis"] >= 2) & (~sp_["c"].isin(test_codes))].copy()
    return sp_, pool


FEATURES = None


def build_history(tx, frame, prod):
    """Historique tronque a t0 : dict par client (retrieval) + tables agregees
    pre-jointes (features, sans aucune boucle Python)."""
    t0 = dict(zip(frame["c"].values, frame["t0"].values))
    sub = tx[tx["c"].isin(t0.keys())]
    sub = sub[sub["day"].values <= sub["c"].map(t0).values]
    sub = sub.sort_values(["c", "day"], kind="mergesort")

    H = {}
    for c, g in sub.groupby("c", sort=False):
        H[int(c)] = (g["p"].values.astype("int32"), g["day"].values.astype("int32"),
                     g["s"].values.astype("int32"))

    s = sub.copy()
    for k in ["f2", "f1", "cat", "uni"]:
        s[k] = prod[k][s["p"].values]
    tt = pd.Series(t0)

    cp = s.groupby(["c", "p"], as_index=False).agg(it_bought_n=("day", "size"), _m=("day", "max"))
    cp["it_days_since"] = (tt.reindex(cp["c"].values).values - cp["_m"].values).astype("float32")
    cp = cp.drop(columns=["_m"])

    def by(col, base):
        g = s.groupby(["c", col], as_index=False).agg(**{base + "_n": ("day", "size"), "_m": ("day", "max")})
        g[base + "_days"] = (tt.reindex(g["c"].values).values - g["_m"].values).astype("float32")
        return g.drop(columns=["_m"])

    T = {"cp": cp, "f2": by("f2", "it_fam2"), "f1": by("f1", "it_fam1"),
         "cat": by("cat", "it_cat"), "uni": by("uni", "it_uni")}

    vis = s[["c", "day"]].drop_duplicates().sort_values(["c", "day"], kind="mergesort")
    dd = vis["day"].diff()
    dd[vis["c"].values != vis["c"].shift().values] = np.nan
    gg = vis.assign(g=dd).groupby("c")["g"].agg(["mean", "std", "median"]).fillna(0.0)
    st = s.groupby("c").agg(cl_n_tx=("p", "size"), cl_n_prod=("p", "nunique"),
                            cl_spend=("amt", "sum"), cl_avg_amt=("amt", "mean"),
                            cl_max_amt=("amt", "max"), cl_min_amt=("amt", "min"),
                            cl_n_cat=("cat", "nunique"), cl_n_f2=("f2", "nunique"),
                            cl_n_stores=("s", "nunique"), cl_avg_qty=("qty", "mean"),
                            _f=("day", "min"), _l=("day", "max"))
    st["cl_n_visits"] = vis.groupby("c").size()
    st["cl_gap_mean"], st["cl_gap_std"], st["cl_gap_med"] = gg["mean"], gg["std"], gg["median"]
    st["cl_tenure"] = st["_l"] - st["_f"]
    st["cl_basket"] = st["cl_n_tx"] / np.maximum(st["cl_n_visits"], 1)
    st["cl_freq"] = st["cl_n_visits"] / (1.0 + st["cl_tenure"] / 30.0)
    st["cl_repeat_rate"] = 1.0 - st["cl_n_prod"] / np.maximum(st["cl_n_tx"], 1)
    st["cl_explore"] = st["cl_n_f2"] / np.maximum(st["cl_n_prod"], 1)
    st["cl_gap_cv"] = st["cl_gap_std"] / (1.0 + st["cl_gap_mean"])
    T["stats"] = st.drop(columns=["_f", "_l"]).reset_index()
    # NOUVEAU v5 - prix unitaire moyen paye par le client DANS CHAQUE CATEGORIE
    s["_unit"] = s["amt"].values / np.maximum(s["qty"].values, 1)
    T["cat_price"] = s.groupby(["c", "cat"], as_index=False)["_unit"].mean().rename(
        columns={"_unit": "cl_cat_price"})

    # magasin de la derniere visite et magasin prefere (+ fidelite au magasin)
    lastst = s.sort_values(["c", "day"], kind="mergesort").groupby("c")["s"].last()
    fs = s.groupby(["c", "s"], as_index=False).size().sort_values(["c", "size"], ascending=[True, False])
    fs1 = fs[fs.groupby("c").cumcount() == 0].set_index("c")
    T["store_last"] = lastst.astype("int32").to_dict()
    T["store_fav"] = fs1["s"].astype("int32").to_dict()
    T["store_loyal"] = (fs1["size"] / st["cl_n_tx"]).astype("float32")
    return H, T


# =============================================================================
# 4. RETRIEVAL : 9 generateurs -> ~650 candidats par client
#    L'objectif ici est le RAPPEL, pas la precision : il vaut mieux 650
#    candidats contenant la reponse que 100 candidats bien choisis mais faux.
# =============================================================================
SRC_NAMES = ["repeat", "popctry3", "popctry12", "popglob", "popseg", "covseq",
             "covbsk", "uu", "famtrans", "family", "category", "storelast",
             "storefav", "svd", "nxt"]
N_META = 10  # src_mask, best_rank, covseq, covbsk, uu, svd, fam, famtrans, covlast, nxt


def _candidates(world, frame, H, T, UU):
    ref = world.ref
    prod, ccountry, ccat = ref["prod"], ref["ccountry"], ref["ccat"]
    SL, SF = T["store_last"], T["store_fav"]
    cols = {k: [] for k in ["c", "p", "src", "rank", "covseq", "covbsk", "uu", "svd", "fam", "ftr",
                            "covlast", "nxt"]}

    for c, t0, mi in zip(frame["c"].values, frame["t0"].values, frame["mi0"].values):
        c = int(c)
        h = H.get(c)
        hp = h[0] if h is not None else np.zeros(0, "int32")
        hd = h[1] if h is not None else np.zeros(0, "int32")
        ci = int(max(ccountry[c], 0))
        seg = int(max(ccat["ClientSegment"][c], 0))

        seeds = []
        if len(hp):
            seen = set()
            for i in np.argsort(-hd, kind="mergesort"):
                q = int(hp[i])
                if q not in seen:
                    seen.add(q)
                    seeds.append(q)

        pool = {}

        def add(ps, bit, scores=None, slot=None, kmax=None):
            n = len(ps) if kmax is None else min(kmax, len(ps))
            for r in range(n):
                q = int(ps[r])
                if q < 0:
                    continue
                e = pool.get(q)
                if e is None:
                    e = [0, 999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    pool[q] = e
                e[0] |= bit
                if r < e[1]:
                    e[1] = r
                if slot is not None and scores is not None and scores[r] > e[slot]:
                    e[slot] = float(scores[r])

        # -- S1 rachat : le signal dominant (plafond 0.2071, atteint a 96 %)
        add(seeds, 1)
        # -- S2/S3 popularite pays, deux fenetres temporelles
        add(world.top_pop(mi, ci, 3, CFG.K_POP_COUNTRY), 2)
        add(world.top_pop(mi, ci, 12, CFG.K_POP_COUNTRY_LONG), 4)
        # -- S4/S5 popularite globale et par segment (couverture)
        add(world.top_pop(mi, 0, 3, CFG.K_POP_GLOBAL, "G"), 8)
        add(world.top_pop(mi, seg, 6, CFG.K_POP_SEGMENT, "S"), 16)
        # -- S12/S13 assortiment magasin : dernier magasin d'abord
        sl = SL.get(c, -1)
        sf = SF.get(c, -1)
        if sl >= 0:
            add(world.STORE_TOP.get(int(sl), np.zeros(0, "int32")), 2048, kmax=CFG.K_STORE_LAST)
        if sf >= 0 and sf != sl:
            add(world.STORE_TOP.get(int(sf), np.zeros(0, "int32")), 4096, kmax=CFG.K_STORE_FAV)

        sd = seeds[:CFG.N_HIST_SEEDS]
        # -- S6 co-visitation sequentielle : "quel produit vient apres"
        agg = defaultdict(float)
        for r, q in enumerate(sd):
            e = world.COV_SEQ.get(q)
            if e is None:
                continue
            b, w = e
            for j in range(min(CFG.K_COVIS_SEQ, len(b))):
                agg[int(b[j])] += float(w[j]) / (1.0 + r)
        if agg:
            it = sorted(agg.items(), key=lambda z: -z[1])[:CFG.K_COVIS_SEQ_OUT]
            add([x[0] for x in it], 32, [x[1] for x in it], 2)
        # -- S6bis (v5) transitions STRICTES depuis les 3 derniers produits, et
        #    score de co-visitation du DERNIER produit seul (non dilue)
        if sd:
            e = world.COV_SEQ.get(sd[0])
            if e is not None:          # score du dernier produit seul, sans toucher au rang
                b, w = e
                for j in range(min(CFG.K_COVIS_SEQ, len(b))):
                    q2 = int(b[j])
                    if q2 in pool and float(w[j]) > pool[q2][8]:
                        pool[q2][8] = float(w[j])
            agg = defaultdict(float)
            for r, q in enumerate(sd[:3]):
                e = world.COV_NXT.get(q)
                if e is None:
                    continue
                b, w = e
                for j in range(min(CFG.K_NXT, len(b))):
                    agg[int(b[j])] += float(w[j]) / (1.0 + r)
            if agg:
                it = sorted(agg.items(), key=lambda z: -z[1])[:CFG.K_NXT_OUT]
                add([x[0] for x in it], 16384, [x[1] for x in it], 9)
        # -- S7 co-occurrence panier
        agg = defaultdict(float)
        for r, q in enumerate(sd[:10]):
            e = world.COV_BSK.get(q)
            if e is None:
                continue
            b, w = e
            for j in range(min(CFG.K_COVIS_BSK, len(b))):
                agg[int(b[j])] += float(w[j]) / (1.0 + r)
        if agg:
            it = sorted(agg.items(), key=lambda z: -z[1])[:CFG.K_COVIS_BSK_OUT]
            add([x[0] for x in it], 64, [x[1] for x in it], 3)
        # -- S8 CF user-user : produits des clients proches dans l'espace SVD
        nb = UU.get(c)
        if nb is not None:
            agg = defaultdict(float)
            ids, sims = nb
            for j in range(len(ids)):
                pr_ = world.prods_of.get(int(ids[j]))
                if pr_ is None:
                    continue
                w = float(sims[j])
                for q in pr_[:25]:
                    agg[int(q)] += w
            if agg:
                it = sorted(agg.items(), key=lambda z: -z[1])[:CFG.K_UU]
                add([x[0] for x in it], 128, [x[1] for x in it], 4)
        # -- S9 transitions de famille : quelle FamilyLevel2 suit quelle autre
        agg = defaultdict(float)
        for r, q in enumerate(sd[:6]):
            f = int(prod["f2"][q])
            u = int(max(prod["uni"][q], 0))
            probs = world.FTRANS_P.get(f, {})
            for nf in world.FTRANS.get(f, [])[:5]:
                e = world.FAM.get(int(nf) * 4 + u)
                if e is None:
                    continue
                pr_ = probs.get(int(nf), 0.0)
                for q2 in e[0][:22]:
                    agg[int(q2)] = max(agg[int(q2)], pr_ / (1.0 + r))
        if agg:
            it = sorted(agg.items(), key=lambda z: -z[1])[:CFG.K_FAMTRANS]
            add([x[0] for x in it], 256, [x[1] for x in it], 7)
        # -- S10/S11 top des familles et categories deja achetees
        fk, ck = [], []
        for q in sd[:6]:
            fk.append(int(prod["f2"][q]) * 4 + int(max(prod["uni"][q], 0)))
            ck.append(int(prod["cat"][q]))
        for r, k in enumerate(dict.fromkeys(fk)):
            e = world.FAM.get(k)
            if e is not None:
                add(e[0], 512, e[1] / (1.0 + r), 6, kmax=CFG.K_FAMILY)
        for k in dict.fromkeys(ck):
            e = world.CATP.get(k)
            if e is not None:
                add(e, 1024, kmax=CFG.K_FAMILY // 2)
        # -- S14 voisins SVD (substituts implicites)
        agg = defaultdict(float)
        for r, q in enumerate(sd[:8]):
            rm = world.svd_remap[q]
            if rm < 0:
                continue
            for j in range(CFG.SVD_NN):
                agg[int(world.svd_nn_i[rm, j])] += float(world.svd_nn_s[rm, j]) / (1.0 + r)
        if agg:
            it = sorted(agg.items(), key=lambda z: -z[1])[:CFG.K_SVD]
            add([x[0] for x in it], 8192, [x[1] for x in it], 5)

        if not pool:
            add(world.top_pop(mi, ci, 3, 30), 2)
            if not pool:
                add(world.top_pop(mi, 0, 3, 30, "G"), 8)

        cand = np.fromiter(pool.keys(), dtype="int32", count=len(pool))
        meta = np.array([pool[int(q)] for q in cand], dtype="float32")
        cols["c"].append(np.full(len(cand), c, dtype="int32"))
        cols["p"].append(cand)
        for i, k in enumerate(["src", "rank", "covseq", "covbsk", "uu", "svd", "fam", "ftr",
                               "covlast", "nxt"]):
            cols[k].append(meta[:, i])

    if not cols["c"]:
        return pd.DataFrame()
    return pd.DataFrame({
        "c": np.concatenate(cols["c"]), "p": np.concatenate(cols["p"]),
        "src_mask": np.concatenate(cols["src"]).astype("int32"),
        "src_best_rank": np.concatenate(cols["rank"]),
        "covis_seq": np.concatenate(cols["covseq"]), "covis_bsk": np.concatenate(cols["covbsk"]),
        "uu_score": np.concatenate(cols["uu"]), "svd_score": np.concatenate(cols["svd"]),
        "fam_score": np.concatenate(cols["fam"]), "ftrans_prob": np.concatenate(cols["ftr"]),
        "covis_last": np.concatenate(cols["covlast"]), "nxt_score": np.concatenate(cols["nxt"]),
    })


# =============================================================================
# 5. FEATURES (~115) : produit point-in-time, client RFM, interaction, magasin,
#    rangs intra-client. Tout est vectorise ou joint - aucune boucle Python.
# =============================================================================
def add_features(world, df, frame, T):
    ref = world.ref
    prod, stock, ccountry, ccat = ref["prod"], ref["stock"], ref["ccountry"], ref["ccat"]
    f = frame.set_index("c")
    c, p = df["c"].values, df["p"].values
    t0 = f["t0"].reindex(c).values.astype("int32")
    mi = f["mi0"].reindex(c).values.astype("int32")

    # ---- provenance : quelles sources ont propose ce candidat ---------------
    sm = df["src_mask"].values
    for i, nm in enumerate(SRC_NAMES):
        df["src_" + nm] = ((sm >> i) & 1).astype("int8")
    df["n_sources"] = sum(df["src_" + nm] for nm in SRC_NAMES).astype("int8")

    # ---- produit, point-in-time --------------------------------------------
    ci = np.maximum(ccountry[c], 0).astype("int32")
    seg = np.maximum(ccat["ClientSegment"][c], 0).astype("int32")
    for w in CFG.POP_WINDOWS:
        df[f"pop_g_{w}m"] = world.pop(world.CUM_G, 0, mi, p, w)
        df[f"pop_c_{w}m"] = world.pop(world.CUM_C, ci, mi, p, w)
    df["pop_s_6m"] = world.pop(world.CUM_S, seg, mi, p, 6)
    df["pop_trend"] = df["pop_g_1m"] / (1.0 + df["pop_g_3m"] / 3.0)
    df["pop_trend_c"] = df["pop_c_1m"] / (1.0 + df["pop_c_3m"] / 3.0)
    df["pop_ctry_share"] = df["pop_c_3m"] / (1.0 + df["pop_g_3m"])
    df["pop_log"] = np.log1p(df["pop_g_12m"].values)
    df["n_buyers"] = world.n_buyers[p]
    df["p_repeat_rate"] = world.p_repeat[p]
    df["p_n_stores"] = world.p_n_stores[p]
    df["age_prod_days"] = np.clip(t0 - world.first_day[p], 0, 10000)
    mm = np.clip(mi - world.m_min + 1, 0, world.n_m - 1)
    df["months_since_sold"] = (mm - world.LAST_M[mm, p]).astype("float32")
    df["price"] = prod["price"][p]
    df["logprice"] = prod["logprice"][p]
    df["price_rank_fam"] = prod["price_rank_fam"][p]
    df["price_vs_fam"] = prod["price_vs_fam"][p]
    df["cat"] = prod["cat"][p].astype("int16")
    df["f1"] = prod["f1"][p].astype("int16")
    df["f2"] = prod["f2"][p].astype("int16")
    df["uni"] = prod["uni"][p].astype("int8")
    df["cat_season"] = world.CAT_SEASON[prod["cat"][p].astype("int32"), (mi % 12).astype("int32")]
    # NOUVEAU v5 - saisonnalite du PRODUIT au mois de t0 et au mois suivant
    moy0 = ((mi - 1) % 12).astype("int32")
    df["p_season_0"] = world.P_SEASON[p, moy0]
    df["p_season_1"] = world.P_SEASON[p, (moy0 + 1) % 12]

    # ---- stock : FEATURE, jamais filtre (filtre dur mesure a -28 % de score)
    q = stock[ci, p]
    df["stock_qty"] = q
    df["in_stock"] = (q > 0).astype("int8")
    df["stock_glob"] = (stock[:, p].sum(axis=0) > 0).astype("int8")

    # ---- magasin : dernier magasin visite, puis magasin prefere -------------
    sl = pd.Series(c).map(T["store_last"]).fillna(-1).values.astype("int32")
    sf = pd.Series(c).map(T["store_fav"]).fillna(-1).values.astype("int32")
    for nm, arr in [("last", sl), ("fav", sf)]:
        ok = arr >= 0
        v = np.zeros(len(df), dtype="float32")
        v[ok] = world.store_mat[arr[ok], p[ok]]
        df[f"store_{nm}_sales"] = v
        df[f"store_{nm}_share"] = v / np.where(ok, world.store_tot[np.maximum(arr, 0)], 1.0)
        df[f"store_{nm}_has"] = (v > 0).astype("int8")
    # part du produit DANS sa famille : chez son magasin, dans son pays, et
    # globalement. Discrimine entre references d'une meme FamilyLevel2.
    fk = world.fam_key[p]
    okl = sl >= 0
    denom = np.ones(len(df), dtype="float32")
    denom[okl] = world.STORE_FAM[sl[okl], fk[okl]]
    df["store_fam_share"] = df["store_last_sales"].values / (1.0 + denom)
    df["ctry_fam_share"] = df["pop_c_12m"].values / (1.0 + world.CTRY_FAM[ci, fk])
    df["glob_fam_share"] = df["pop_g_12m"].values / (1.0 + world.GLOB_FAM[fk])
    df["fam_size"] = world.GLOB_FAM[fk]
    df["store_same"] = (sl == sf).astype("int8")
    # NOUVEAU v5 - anciennete du produit DANS le dernier magasin du client
    sfirst = np.full(len(df), 10 ** 6, dtype="int32")
    sfirst[okl] = world.STORE_FIRST[sl[okl], p[okl]]
    age_s = np.where(sfirst < 10 ** 6, t0 - sfirst, -1).astype("float32")
    df["store_p_age"] = np.clip(age_s, -1, 10000)
    df["store_p_new60"] = ((age_s >= 0) & (age_s <= 60)).astype("int8")
    df["cl_store_loyal"] = pd.Series(c).map(T["store_loyal"]).fillna(0).values.astype("float32")

    # ---- client + interactions, par jointures -------------------------------
    df = df.merge(T["stats"], on="c", how="left")
    df = df.merge(T["cp"], on=["c", "p"], how="left")
    for key, col in [("f2", "f2"), ("f1", "f1"), ("cat", "cat"), ("uni", "uni")]:
        df = df.merge(T[key], on=["c", col], how="left")
    df["it_bought_n"] = df["it_bought_n"].fillna(0).astype("float32")
    df["it_days_since"] = df["it_days_since"].fillna(9999).astype("float32")
    for nm in ["it_fam2_n", "it_fam1_n", "it_cat_n", "it_uni_n"]:
        df[nm] = df[nm].fillna(0).astype("float32")
    for nm in ["it_fam2_days", "it_fam1_days", "it_cat_days", "it_uni_days"]:
        df[nm] = df[nm].fillna(9999).astype("float32")
    for nm in T["stats"].columns:
        if nm != "c":
            df[nm] = df[nm].fillna(0).astype("float32")

    ntx = np.maximum(df["cl_n_tx"].values, 1)
    df["it_fam2_share"] = df["it_fam2_n"] / ntx
    df["it_fam1_share"] = df["it_fam1_n"] / ntx
    df["it_cat_share"] = df["it_cat_n"] / ntx
    df["it_uni_share"] = df["it_uni_n"] / ntx
    df["it_price_ratio"] = df["price"] / np.maximum(df["cl_avg_amt"], 0.01)
    # NOUVEAU v5 - prix du candidat rapporte a ce que le client paie d'habitude
    # DANS CETTE CATEGORIE (bien plus precis que le panier moyen global)
    df = df.merge(T["cat_price"], on=["c", "cat"], how="left")
    cp_ = df["cl_cat_price"].fillna(df["cl_avg_amt"]).values.astype("float32")
    df["it_cat_price_ratio"] = df["price"].values / np.maximum(cp_, 0.01)
    df["it_cat_price_absdev"] = np.abs(np.log1p(df["price"].values) - np.log1p(cp_))
    df = df.drop(columns=["cl_cat_price"])
    df["it_price_gap_max"] = df["price"] / np.maximum(df["cl_max_amt"], 0.01)
    df["it_recency_score"] = df["it_bought_n"] / (1.0 + df["it_days_since"] / 30.0)
    df["it_cov_x_rep"] = df["covis_seq"] * (1 + df["it_bought_n"])
    # "le client est-il DU pour ce produit / cette famille ?" : delai ecoule
    # rapporte a sa periodicite d'achat propre
    gm = 1.0 + df["cl_gap_mean"].values
    df["it_due_ratio"] = np.clip(df["it_days_since"].values / gm, 0, 50)
    df["it_fam2_due"] = np.clip(df["it_fam2_days"].values / gm, 0, 50)
    df["it_cat_due"] = np.clip(df["it_cat_days"].values / gm, 0, 50)

    df["cl_segment"] = ccat["ClientSegment"][c].astype("int8")
    df["cl_country"] = ccat["ClientCountry"][c].astype("int8")
    df["cl_gender"] = ccat["ClientGender"][c].astype("int8")
    df["cl_age"] = ccat["Age"][c]
    df["cl_opt_email"] = ccat["ClientOptINEmail"][c].astype("int8")

    gcats = ccat["ClientGender_cats"]
    gi = ccat["ClientGender"][c]
    gname = np.array([gcats[i] if 0 <= i < len(gcats) else "NA" for i in gi], dtype=object)
    uw = (prod["uni"][p] == prod["uni_women"]).astype("int8")
    df["it_gender_match"] = (((gname == "F") & (uw == 1)) | ((gname == "M") & (uw == 0))).astype("int8")

    # ---- rangs et parts intra-client : ce que le ranker exploite le mieux ---
    g = df.groupby("c")
    for nm in ["pop_c_3m", "pop_g_3m", "covis_seq", "uu_score", "svd_score", "nxt_score", "covis_last",
               "it_recency_score", "fam_score", "store_last_sales", "ftrans_prob"]:
        df["rk_" + nm] = g[nm].rank(ascending=False, method="first").astype("float32")
    for nm in ["pop_c_3m", "covis_seq", "uu_score", "store_last_sales"]:
        df["shr_" + nm] = df[nm] / (1e-6 + g[nm].transform("sum"))
    df["grp_size"] = g["p"].transform("size").astype("float32")
    gf = df.groupby(["c", "f2"])
    df["rk_in_fam"] = gf["store_fam_share"].rank(ascending=False, method="first").astype("float32")
    df["rk_in_fam_pop"] = gf["pop_c_3m"].rank(ascending=False, method="first").astype("float32")
    df["n_in_fam"] = gf["p"].transform("size").astype("float32")

    # ---- compression memoire : float32 partout -----------------------------
    for col in df.columns:
        if col in ("c", "p", "y"):
            continue
        if df[col].dtype.kind in "iub":
            if df[col].dtype.itemsize > 2:
                df[col] = df[col].astype("int16")
        elif df[col].dtype != np.float32:
            df[col] = df[col].astype("float32")
    df = df.drop(columns=["src_mask"])

    global FEATURES
    if FEATURES is None:
        FEATURES = [x for x in df.columns if x not in ("c", "p", "y")]
    return df


class MatrixSink:
    """Accumule les blocs dans une matrice float32 PREALLOUEE.

    pd.concat double la memoire au moment de l'assemblage : sur un jeu de
    30 M de lignes cela suffit a faire tomber la machine. Ici le pic est
    exactement la taille finale, connue a l'avance (n_clients x cap)."""

    def __init__(self, n_clients, cap_per_client, feats):
        self.cap = int(np.ceil(n_clients * cap_per_client))
        self.feats = feats
        self.X = None
        self.y = np.empty(self.cap, dtype="int8")
        self.c = np.empty(self.cap, dtype="int32")
        self.p = np.empty(self.cap, dtype="int32")
        self.n = 0
        self.dropped = 0

    def push(self, df):
        if self.X is None:
            self.X = np.empty((self.cap, len(self.feats)), dtype="float32")
        k = min(len(df), self.cap - self.n)
        if k < len(df):
            self.dropped += len(df) - k
        if k <= 0:
            return
        self.X[self.n:self.n + k] = df[self.feats].to_numpy(dtype=np.float32, copy=False)[:k]
        self.y[self.n:self.n + k] = (df["y"].values[:k] if "y" in df.columns else 0)
        self.c[self.n:self.n + k] = df["c"].values[:k]
        self.p[self.n:self.n + k] = df["p"].values[:k]
        self.n += k

    def done(self):
        if self.X is None:
            return np.zeros((0, len(self.feats)), "float32"), np.zeros(0, "int8"), \
                   np.zeros(0, "int32"), np.zeros(0, "int32")
        if self.dropped:
            log(f"   ATTENTION : {self.dropped:,} lignes ecartees (cap atteint)")
        return self.X[:self.n], self.y[:self.n], self.c[:self.n], self.p[:self.n]


def sample_negatives(df, n_neg, rng):
    """Tous les positifs + n_neg negatifs tires au hasard par groupe.
    Les groupes sans positif sont jetes : ils n'apportent aucun signal a
    LambdaRank. IMPORTANT : cet echantillonnage intervient APRES le calcul des
    features de rang, qui doivent voir le pool complet (sinon le rang vu a
    l'entrainement n'a rien a voir avec celui vu en production)."""
    y, c = df["y"].values, df["c"].values
    df = df[pd.Series(y).groupby(c).transform("max").values > 0]
    if len(df) == 0:
        return df
    y, c = df["y"].values, df["c"].values
    r = rng.random(len(df))
    r[y == 1] = -1.0
    o = np.lexsort((r, c))
    rnk = np.empty(len(df), dtype="int32")
    ser = pd.Series(c[o])
    rnk[o] = ser.groupby(ser.values).cumcount().values
    npos = pd.Series(y).groupby(c).transform("sum").values
    return df[(y == 1) | (rnk < npos + n_neg)]


def generate(world, frame, H, T, truth=None, n_neg=None, rng=None, desc="", sink=None,
             gid_offset=0):
    """Retrieval -> features -> (echantillonnage). Par blocs de clients pour
    borner la memoire : le pic est celui d'UN bloc, pas de tout le jeu."""
    outs, n = [], len(frame)
    pos = None
    if truth is not None:
        pos = np.sort(np.fromiter(((int(a) << 31) + int(b) for a, ss in truth.items() for b in ss),
                                  dtype="int64"))
    t_start = time.time()
    for i in range(0, n, CFG.CHUNK_CLIENTS):
        fr = frame.iloc[i:i + CFG.CHUNK_CLIENTS]
        UU = world.user_neighbors(fr["c"].values.astype("int32"))
        df = _candidates(world, fr, H, T, UU)
        del UU
        if len(df) == 0:
            continue
        if truth is not None:
            key = (df["c"].values.astype("int64") << 31) + df["p"].values
            df["y"] = np.isin(key, pos).astype("int8")
            del key
        df = add_features(world, df, fr, T)
        if truth is not None and n_neg is not None:
            df = sample_negatives(df, n_neg, rng)
        if sink is not None:
            if gid_offset:
                # un meme client peut fournir plusieurs instantanes : on decale
                # son identifiant de groupe pour que LightGBM voie des groupes
                # distincts et contigus (l'identifiant reel n'est plus utile ici)
                df["c"] = (df["c"].values.astype("int64") + gid_offset).astype("int32")
            sink.push(df)
            del df
        else:
            outs.append(df)
        if desc and ((i // CFG.CHUNK_CLIENTS) % 3 == 0 or i + CFG.CHUNK_CLIENTS >= n):
            done = min(i + CFG.CHUNK_CLIENTS, n)
            el = time.time() - t_start
            log(f"   {desc} {done:,}/{n:,} clients  ({el:.0f}s, ETA {el / done * (n - done):.0f}s)")
        gc.collect()
    if sink is not None:
        return sink.done()
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# =============================================================================
# 6. EVALUATION - reimplemente exactement le Recall@5 officiel
# =============================================================================
def recall_at_k(topk, truth, k=5):
    """Retourne (HIT RATE@k, Recall@k).

    ATTENTION - la metrique officielle du hackathon est le HIT RATE@5 : 1 si au
    moins un produit de la visite cible figure dans les 5 propositions, 0 sinon.
    C'est la seule metrique d'evaluation. Le Recall@5 (part des produits de la
    visite retrouves) est conserve comme diagnostic.

    Les deux ne different que sur les visites a plusieurs produits, soit 7.8 %
    des clients en population calibree (92.2 % des visites cibles ne
    contiennent qu'un seul produit). Mesure de l'ecart :
        visite a 1 produit : Recall 0.2410 = HitRate 0.2410
        visite a 2 produits: Recall 0.1756 -> HitRate 0.2849
        visite a 3 produits: Recall 0.1347 -> HitRate 0.2560
    Le Hit Rate est donc structurellement superieur de ~3.5 % au Recall.
    Toute optimisation et tout arbitrage doivent se faire sur le HIT RATE."""
    ks = [c for c in truth if c in topk]
    hit = [1.0 if set(topk[c][:k]) & truth[c] else 0.0 for c in ks]
    rec = [len(set(topk[c][:k]) & truth[c]) / len(truth[c]) for c in ks]
    return float(np.mean(hit)), float(np.mean(rec))


def top_k(df, score, k=5):
    d = pd.DataFrame({"c": df["c"].values, "p": df["p"].values, "s": score})
    d = d.sort_values(["c", "s"], ascending=[True, False], kind="mergesort")
    d = d[d.groupby("c").cumcount() < k]
    return {int(a): list(b) for a, b in d.groupby("c")["p"].apply(list).items()}


def top_k_diverse(df, score, world, lam, k=5, basket_size=None):
    """Selection gloutonne des 5 produits sous metrique HIT RATE.

    Sous Recall@5, proposer deux produits souvent achetes ENSEMBLE rapporte
    deux fois. Sous Hit Rate@5, cela ne rapporte qu'une fois : si le client
    achete les deux, on ne marque qu'un point. L'optimum n'est donc plus les 5
    probabilites les plus elevees mais l'ensemble qui maximise
    P(au moins un bon) - un probleme de couverture, resolu ici par un glouton.

    A chaque etape on retient le meilleur candidat restant, puis on penalise
    les candidats qui co-apparaissent dans les memes paniers que ceux deja
    retenus (matrice COV_BSK). Les SUBSTITUTS (memes alternatives, rarement
    achetees ensemble) ne sont pas penalises : ce sont des paris mutuellement
    exclusifs, donc additifs, et il faut les garder.

    lam = 0 redonne exactement la selection classique.
    basket_size : panier moyen historique du client. La penalite n'a de sens
    que pour les clients susceptibles d'acheter plusieurs produits ; elle est
    donc modulee par ce facteur."""
    if lam <= 0:
        return top_k(df, score, k)
    c = df["c"].values
    p = df["p"].values
    d = pd.DataFrame({"c": c, "p": p, "s": score})
    d = d.sort_values(["c", "s"], ascending=[True, False], kind="mergesort")
    # on ne travaille que sur le haut de liste : au-dela le glouton ne change rien
    d = d[d.groupby("c").cumcount() < 40]
    out = {}
    BSK = world.COV_BSK
    for cid, g in d.groupby("c", sort=False):
        cand = g["p"].values
        sc = g["s"].values.astype("float64").copy()
        mult = 1.0
        if basket_size is not None:
            mult = float(np.clip(basket_size.get(int(cid), 1.0) - 1.0, 0.0, 2.0))
            if mult <= 0:
                out[int(cid)] = list(cand[:k])
                continue
        chosen, used = [], np.zeros(len(cand), dtype=bool)
        pen = np.zeros(len(cand), dtype="float64")
        for _ in range(min(k, len(cand))):
            v = sc - lam * mult * pen
            v[used] = -1e18
            j = int(np.argmax(v))
            used[j] = True
            chosen.append(int(cand[j]))
            e = BSK.get(int(cand[j]))
            if e is not None:
                b, w = e
                mp = dict(zip(b.tolist(), w.tolist()))
                mx = max(mp.values()) if mp else 1.0
                for i2 in range(len(cand)):
                    if not used[i2]:
                        pen[i2] += mp.get(int(cand[i2]), 0.0) / (mx + 1e-9)
        out[int(cid)] = chosen
    return out


def group_sizes(a):
    """Tailles de groupes sur runs contigus : les candidats sortent deja
    groupes par client, donc aucun tri (donc aucune copie) n'est necessaire."""
    b = np.flatnonzero(np.r_[True, a[1:] != a[:-1], True])
    g = np.diff(b)
    assert len(g) == len(np.unique(a)), "candidats non contigus par client"
    return g


# =============================================================================
# 7. ETAGE 2 -> ELAGAGE -> ETAGE 3
#    On genere le pool complet par bloc, on le score avec le ranker 1 et on ne
#    garde que les PRUNE_TO meilleurs. Le pic memoire est celui d'un bloc, et
#    le ranker 2 voit exactement le meme format a l'entrainement et en
#    production (meme taille de groupe) - pas de decalage train/serve.
# =============================================================================
CATS = ["cat", "f1", "f2", "uni", "cl_segment", "cl_country", "cl_gender"]


def predict_rank(models, D, feats, block=1_000_000):
    """Moyenne des rangs normalises intra-client sur l'ensemble des modeles."""
    acc = np.zeros(len(D), dtype="float32")
    for m in models:
        out = np.empty(len(D), dtype="float32")
        ni = m.best_iteration or 0
        for i in range(0, len(D), block):
            j = min(i + block, len(D))
            out[i:j] = m.predict(D[feats].iloc[i:j], num_iteration=ni).astype("float32")
        acc += pd.Series(out).groupby(D["c"].values).rank(pct=True).values.astype("float32")
    return acc / max(len(models), 1)


def generate_rank_prune(world, frame, H, T, models1, prune_to,
                        truth=None, diag=None, desc="", sink=None, pos_only=False):
    outs, n = [], len(frame)
    pos = None
    if truth is not None:
        pos = np.sort(np.fromiter(((int(a) << 31) + int(b) for a, ss in truth.items() for b in ss),
                                  dtype="int64"))
    t_start = time.time()
    for i in range(0, n, CFG.CHUNK_CLIENTS):
        fr = frame.iloc[i:i + CFG.CHUNK_CLIENTS]
        UU = world.user_neighbors(fr["c"].values.astype("int32"))
        df = _candidates(world, fr, H, T, UU)
        del UU
        if len(df) == 0:
            continue
        if truth is not None:
            key = (df["c"].values.astype("int64") << 31) + df["p"].values
            df["y"] = np.isin(key, pos).astype("int8")
            del key
        df = add_features(world, df, fr, T)

        s1 = predict_rank(models1, df, FEATURES)
        d = pd.DataFrame({"i": np.arange(len(df)), "c": df["c"].values, "s": s1})
        d = d.sort_values(["c", "s"], ascending=[True, False], kind="mergesort")
        d["r"] = d.groupby("c").cumcount()

        if diag is not None and truth is not None:      # diagnostics etage 1+2
            y = df["y"].values
            nt = pd.Series({k: len(v) for k, v in truth.items()})
            hit_all = pd.Series(y).groupby(df["c"].values).sum()
            diag["cand"].append(hit_all.values / nt.reindex(hit_all.index.values).values)
            yy = y[d["i"].values]
            for K in diag["Ks"]:
                sel = d["r"].values < K
                hk = pd.Series(yy[sel]).groupby(d["c"].values[sel]).sum()
                diag[f"r{K}"].append(hk.values / nt.reindex(hk.index.values).values)

        keep = d[d["r"] < prune_to]
        sub = df.iloc[keep["i"].values].copy()
        sub["s1_score"] = keep["s"].values.astype("float32")
        sub["s1_rank"] = keep["r"].values.astype("float32")
        sub = sub.sort_values("c", kind="mergesort")
        if pos_only and "y" in sub.columns:
            # jeter tout de suite les groupes sans positif : ils n'apportent
            # aucun gradient a LambdaRank et couteraient une copie complete
            m = pd.Series(sub["y"].values).groupby(sub["c"].values).transform("max").values > 0
            sub = sub[m]
        if sink is not None:
            sink.push(sub)
            del sub
        else:
            outs.append(sub)
        del df, d, keep
        if desc and ((i // CFG.CHUNK_CLIENTS) % 3 == 0 or i + CFG.CHUNK_CLIENTS >= n):
            done = min(i + CFG.CHUNK_CLIENTS, n)
            el = time.time() - t_start
            log(f"   {desc} {done:,}/{n:,}  ({el:.0f}s, ETA {el / done * (n - done):.0f}s)")
        gc.collect()
    if sink is not None:
        return sink.done()
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def train_ranker(D_tr, D_es, feats, params, rounds, n_seeds, tag, mdl_dir):
    """D_tr / D_es : DataFrame, ou tuple (X, y, c, p) issu d'un MatrixSink."""
    def ds(D, ref=None):
        if isinstance(D, tuple):
            X, y, c, _ = D
        else:
            X = D[feats].to_numpy(dtype=np.float32, copy=False)
            y, c = D["y"].to_numpy(), D["c"].values
        o = lgb.Dataset(X, label=y, group=group_sizes(c), feature_name=list(feats),
                        categorical_feature=[x for x in CATS if x in feats],
                        reference=ref, free_raw_data=True)
        o.construct()
        del X
        gc.collect()
        return o

    dtr = ds(D_tr)
    dva = ds(D_es, dtr)
    models = []
    for s in range(n_seeds):
        pr = dict(params)
        pr.update({"seed": CFG.SEED + s * 101, "bagging_seed": CFG.SEED + s * 7,
                   "feature_fraction_seed": CFG.SEED + s * 13})
        # Ensemble MULTI-OBJECTIFS. Les trois objectifs donnent le meme score
        # seul (0.2658 / 0.2665 / cf. journal) mais se trompent sur des clients
        # DIFFERENTS : moyenner leurs rangs reduit la variance, ce qui compte
        # double quand le score final se joue sur 200 clients (+/- 5.5 pts).
        if CFG.MULTI_OBJ and tag == "s1":
            o = CFG.MULTI_OBJ[s % len(CFG.MULTI_OBJ)]
            if o == "binary":
                pr.update({"objective": "binary", "metric": "binary_logloss"})
                pr.pop("ndcg_eval_at", None)
                pr.pop("lambdarank_truncation_level", None)
            else:
                pr.update({"objective": o, "metric": "ndcg", "ndcg_eval_at": [5]})
                pr.setdefault("lambdarank_truncation_level", 25)
            tag_s = f"{tag}[{o}]"
        else:
            tag_s = tag
        with timer(f"LightGBM {tag_s} seed {s + 1}/{n_seeds}"):
            m = lgb.train(pr, dtr, num_boost_round=rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(CFG.EARLY_STOP, verbose=False),
                                     lgb.log_evaluation(200)])
            _k = list(m.best_score["valid_0"])[0]
            m.save_model(os.path.join(mdl_dir, f"{tag}_{s}.txt"), num_iteration=m.best_iteration)
            models.append(m)
            log(f"   best_iter={m.best_iteration}  {_k}={m.best_score['valid_0'][_k]:.5f}")
    del dtr, dva
    gc.collect()
    return models


# =============================================================================
# 8. PIPELINE
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-train2", type=int)
    ap.add_argument("--n-valid", type=int)
    ap.add_argument("--n-neg", type=int)
    ap.add_argument("--prune", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--n-es", type=int)
    ap.add_argument("--kstore", type=int, help="taille de l'assortiment magasin retenu")
    ap.add_argument("--folds", type=int, help="nb de folds cross-fittes (defaut 2)")
    ap.add_argument("--bags", type=int, help="nb de bags (1 = pipeline v4 avec etage 3 ; defaut 3)")
    ap.add_argument("--snapshots", type=int, help="instantanes par client (1 = comme v4, defaut 2)")
    ap.add_argument("--no-match", action="store_true",
                    help="evalue sur population uniforme (CV non comparable au leaderboard)")
    ap.add_argument("--obj", choices=["lambdarank", "xendcg", "binary"], default=None,
                    help="force un objectif unique pour l'etage 2 (desactive l'ensemble)")
    ap.add_argument("--match-train", action="store_true",
                    help="calibre aussi l'entrainement (mesure : -3.4 %%, deconseille)")
    args = ap.parse_args()

    if args.fast:
        CFG.N_TRAIN_CLIENTS, CFG.N_TRAIN2_CLIENTS, CFG.N_VALID_CLIENTS = 10_000, 6_000, 4_000
        CFG.N_ES_CLIENTS = 2_000
        CFG.N_NEG_PER_CLIENT, CFG.PRUNE_TO = 40, 100
        CFG.ROUNDS1 = CFG.ROUNDS2 = 300
        CFG.N_SEEDS1 = CFG.N_SEEDS2 = 1
        CFG.SVD_DIM, CFG.EARLY_STOP = 48, 50
        CFG.CHUNK_CLIENTS = 2_000
        CFG.N_FOLDS, CFG.N_SNAPSHOTS = 2, 2
        CFG.N_BAGS = 2
    if args.max:
        # Pleine puissance : 6 bags x 110k clients, 120 negatifs par groupe,
        # learning rate faible. Le pool de candidats n'est PAS elargi (mesure
        # deux fois : l'elargir fait baisser le score).
        #
        # RAM  : pic ~13 Go (un bag a 2 instantanes : matrice 8.6 Go + copie
        #        binnee LightGBM 1.9 Go + monde 1.5 Go). Machine 32 Go : ok.
        #        Machine 16 Go : ajouter --n-neg 90 --chunk 1500 (pic ~10 Go).
        # TEMPS: par bag, generation 6-15 min (Python mono-thread), LightGBM
        #        8-20 min selon les coeurs, validation 3-5 min. Six bags :
        #        2.5-3.5 h sur 8+ coeurs, 5-6 h sur 4 coeurs.
        CFG.N_TRAIN_CLIENTS, CFG.N_TRAIN2_CLIENTS, CFG.N_VALID_CLIENTS = 110_000, 0, 20_000
        CFG.N_ES_CLIENTS = 5_000
        CFG.N_NEG_PER_CLIENT, CFG.PRUNE_TO = 120, 140
        CFG.ROUNDS1, CFG.ROUNDS2, CFG.EARLY_STOP = 4000, 4000, 250
        CFG.N_SEEDS1, CFG.N_SEEDS2 = 1, 1
        CFG.N_BAGS, CFG.K_FUSE = 6, 30
        CFG.SVD_DIM, CFG.SVD_NN, CFG.UU_K = 128, 26, 70
        CFG.CHUNK_CLIENTS = 2_500
        CFG.LGB1.update({"learning_rate": 0.015, "num_leaves": 255,
                         "min_data_in_leaf": 60, "lambda_l2": 5.0})
        CFG.LGB2.update({"learning_rate": 0.02, "num_leaves": 160,
                         "min_data_in_leaf": 80, "lambda_l2": 6.0,
                         "lambdarank_truncation_level": 10})
    if args.n_train2 == 0:
        CFG.N_TRAIN2_CLIENTS = 0
    for k, v in [("N_TRAIN_CLIENTS", args.n_train), ("N_TRAIN2_CLIENTS", args.n_train2),
                 ("N_VALID_CLIENTS", args.n_valid), ("N_NEG_PER_CLIENT", args.n_neg),
                 ("PRUNE_TO", args.prune), ("CHUNK_CLIENTS", args.chunk),
                 ("N_ES_CLIENTS", args.n_es)]:
        if v:
            setattr(CFG, k, v)
    if args.rounds:
        CFG.ROUNDS1 = CFG.ROUNDS2 = args.rounds
    if args.seeds:
        CFG.N_SEEDS1 = CFG.N_SEEDS2 = args.seeds
    if args.folds:
        CFG.N_FOLDS = args.folds
    if args.bags:
        CFG.N_BAGS = args.bags
    if args.snapshots:
        CFG.N_SNAPSHOTS = args.snapshots
    if args.kstore:
        CFG.K_STORE_LAST = args.kstore
        CFG.K_STORE_FAV = max(80, args.kstore // 2)
    if args.obj:
        CFG.MULTI_OBJ = None
    if args.obj == "xendcg":
        CFG.LGB1.update({"objective": "rank_xendcg"})
    elif args.obj == "binary":
        # Argument theorique : pour une cible a UN seul produit (87.8 % des
        # visites), E[Recall@5] = somme des P(item) sur les 5 choisis. Un
        # classifieur binaire bien calibre maximise donc directement la
        # metrique, alors que LambdaRank optimise NDCG, qui sur-pondere la
        # position 1 - un ordre optimal pour NDCG ne l'est pas pour Recall@5.
        CFG.LGB1.update({"objective": "binary", "metric": "binary_logloss"})
        CFG.LGB1.pop("ndcg_eval_at", None)
        CFG.LGB1.pop("lambdarank_truncation_level", None)
    if args.no_match:
        CFG.MATCH_EVAL = False
    if args.match_train:
        CFG.MATCH_TRAIN = True
    if args.workers:
        CFG.LGB1["num_threads"] = CFG.LGB2["num_threads"] = args.workers

    rng = np.random.default_rng(CFG.SEED)
    t_start = time.time()
    mdl_dir = os.path.join(CFG.CACHE_DIR, "models2")
    os.makedirs(mdl_dir, exist_ok=True)

    with timer("chargement"):
        tx, clients, products, stores, stocks, tval, tfin = load_raw()
        log(f"   {len(tx):,} transactions | {len(clients):,} clients | {len(products):,} produits")
    tx, ref = prepare(tx, clients, products, stocks)
    cv = ref["cv"]
    tv_c, tf_c = cv.enc(tval["ClientID"]), cv.enc(tfin["ClientID"])
    test_c = np.concatenate([tv_c, tf_c])

    split, pool = build_split(tx, set(test_c.tolist()))
    allp = pool["c"].values.copy()
    rng.shuffle(allp)
    tset = set(test_c.tolist())
    n_va, n_es = CFG.N_VALID_CLIENTS, CFG.N_ES_CLIENTS
    # --- evaluation + early stopping : population CALIBREE (representativite)
    if CFG.MATCH_EVAL:
        ev_ids = match_test_profile(allp, split, tset, n_va + n_es, rng)
        log("evaluation CALIBREE sur le profil des clients de test "
            "-> le Recall@5 affiche est comparable au leaderboard")
    else:
        ev_ids = allp[:n_va + n_es]
    n_va = min(n_va, max(1, len(ev_ids) - n_es))
    va_ids, es_ids = ev_ids[:n_va], ev_ids[n_va:n_va + n_es]
    # --- entrainement : TOUT le pool restant (volume), sauf si MATCH_TRAIN
    rest = np.setdiff1d(allp, ev_ids)
    rng.shuffle(rest)
    if CFG.MATCH_TRAIN:
        tr_ids = match_test_profile(rest, split, tset, CFG.N_TRAIN_CLIENTS, rng)
        log("entrainement CALIBRE (deconseille : -3.4 % mesure)")
    else:
        tr_ids = rest[:CFG.N_TRAIN_CLIENTS]
    tr2_ids = tr_ids[:min(CFG.N_TRAIN2_CLIENTS, len(tr_ids))]
    log(f"clients : etage2={len(tr_ids):,}  etage3={len(tr2_ids):,}  "
        f"early-stop={len(es_ids):,}  valid={len(va_ids):,}  test={len(test_c)}")
    log(f"config : candidats~{CFG.K_POP_COUNTRY + CFG.K_STORE_LAST + CFG.K_COVIS_SEQ_OUT + CFG.K_UU + CFG.K_FAMTRANS + 250}"
        f"  n_neg={CFG.N_NEG_PER_CLIENT}  prune={CFG.PRUNE_TO}  "
        f"seeds={CFG.N_SEEDS1}/{CFG.N_SEEDS2}")

    # =====================================================================
    # MONDES CROSS-FITTES + ENTRAINEMENT MULTI-INSTANTANES  (nouveau en v5)
    #
    # Avant : un exemple par client (cible = derniere visite), un seul monde
    # prive des dernieres visites de tous les clients d'entrainement.
    # Maintenant : N_SNAPSHOTS exemples par client (derniere visite, puis
    # avant-derniere avec un historique plus court) et N_FOLDS mondes. Le
    # monde du fold f est prive des visites-cibles des clients du fold f
    # SEULEMENT ; les autres folds y restent entiers. Aucune fuite - la
    # transition (dernier produit -> cible) d'un client n'est jamais dans le
    # monde qui sert a construire son exemple - et le volume de donnees
    # d'agregats perdu reste le meme qu'avant.
    # =====================================================================
    sp = split.set_index("c")
    lab_eval = np.concatenate([va_ids, es_ids])
    folds = np.array_split(tr_ids, max(1, CFG.N_FOLDS))
    lastday = sp["last_day"]

    def snap_targets(cids, snap):
        """(clients, date cible, date t0) pour l'instantane `snap`.
        snap 0 : cible = derniere visite, t0 = avant-derniere.
        snap 1 : cible = avant-derniere, t0 = antepenultieme (3 visites requises)."""
        cids = np.asarray(cids, dtype="int32")
        d = sp.reindex(cids)
        if snap == 0:
            ok = d["t0"].notna().values
            return cids[ok], d["last_day"].values[ok].astype("int32"), d["t0"].values[ok].astype("int32")
        ok = d["t0b"].notna().values
        return cids[ok], d["t0"].values[ok].astype("int32"), d["t0b"].values[ok].astype("int32")

    def frame_at(cids, t0):
        t0 = np.asarray(t0, dtype="int32")
        day0 = pd.Timestamp("2023-01-01") + pd.to_timedelta(t0, unit="D")
        return pd.DataFrame({"c": np.asarray(cids, dtype="int32"), "t0": t0,
                             "mi0": (day0.year * 12 + day0.month).values.astype("int32")})

    def truth_of(cids, tdate):
        kt = pd.Series(np.asarray(tdate), index=np.asarray(cids))
        sub = tx[tx["c"].isin(set(np.asarray(cids).tolist()))]
        sub = sub[sub["day"].values == kt.reindex(sub["c"].values).values]
        return sub.groupby("c")["p"].apply(set).to_dict()

    def world_for(fold_ids, tag, n_snap=None):
        """Monde prive des visites-cibles de tous les instantanes des clients du
        fold + de la derniere visite des clients d'evaluation."""
        drop = []
        for snap in range(n_snap or CFG.N_SNAPSHOTS):
            cc, td, _ = snap_targets(fold_ids, snap)
            drop.append(pd.DataFrame({"c": cc, "day": td}))
        cc, td, _ = snap_targets(lab_eval, 0)
        drop.append(pd.DataFrame({"c": cc, "day": td}))
        drop = pd.concat(drop).drop_duplicates()
        k_all = tx["c"].values.astype("int64") * (1 << 20) + tx["day"].values
        k_drop = drop["c"].values.astype("int64") * (1 << 20) + drop["day"].values
        is_t = np.isin(k_all, k_drop)
        tx_f = tx[~is_t]
        log(f"monde {tag} : {len(tx_f):,} transactions ({int(is_t.sum()):,} visites-cibles retirees)")
        return tx_f

    # evaluation : cible = derniere visite (protocole officiel)
    va_c, va_td, va_t0 = snap_targets(va_ids, 0)
    es_c, es_td, es_t0 = snap_targets(es_ids, 0)
    fr_va, fr_es = frame_at(va_c, va_t0), frame_at(es_c, es_t0)
    truth = truth_of(np.concatenate([va_c, es_c]), np.concatenate([va_td, es_td]))
    va_ids = va_c

    def frame_of(cids, use_last=False):
        d = sp.reindex(cids)
        t0 = (d["last_day"] if use_last else d["t0"].fillna(d["last_day"])).values.astype("int32")
        return frame_at(cids, t0)

    # =====================================================================
    # CHOIX DU CHEMIN : bagging (N_BAGS > 1) ou pipeline simple avec etage 3
    # =====================================================================
    if CFG.N_BAGS > 1:
        # -----------------------------------------------------------------
        # BAGGING SUR LES CONFIGURATIONS DE DONNEES
        # Chaque bag : nouvelle partition des clients en folds, alternance
        # 1 / 2 instantanes, ses propres mondes, son propre modele. Sa
        # validation est scoree pendant que son monde est en vie, et seuls
        # ses top-K_FUSE par client sont conserves : la memoire reste celle
        # d'UN bag. Les bags sont fusionnes par vote de Borda.
        # -----------------------------------------------------------------
        log(f"BAGGING : {CFG.N_BAGS} bags x {CFG.N_SEEDS1} seed(s), fusion Borda top-{CFG.K_FUSE}")
        bag_models, bag_tops, bag_scores = [], [], []
        cand_hits = []
        for bag in range(CFG.N_BAGS):
            rng_b = np.random.default_rng(CFG.SEED + 1000 * (bag + 1))
            n_snap = 1 + (bag % 2)                      # alternance 1 / 2 instantanes
            n_fold = 1 if bag == 0 else CFG.N_FOLDS       # bag 0 = configuration v4
            ids_b = tr_ids.copy()
            rng_b.shuffle(ids_b)
            folds_b = np.array_split(ids_b, n_fold)
            n_ex = sum(len(snap_targets(fd, sn)[0]) for fd in folds_b for sn in range(n_snap))
            log(f"--- BAG {bag + 1}/{CFG.N_BAGS} : {n_fold} fold(s), {n_snap} instantane(s), "
                f"{n_ex:,} exemples")
            sink_b, W_b, tx_b = None, None, None
            for f_idx in reversed(range(n_fold)):
                fold = folds_b[f_idx]
                tx_f = world_for(fold, f"bag{bag}-fold{f_idx}", n_snap)
                W_f = World(tx_f, ref, f"bag{bag}-fold{f_idx}")
                for snap in range(n_snap):
                    cc, td, t0s = snap_targets(fold, snap)
                    if len(cc) == 0:
                        continue
                    fr = frame_at(cc, t0s)
                    tru = truth_of(cc, td)
                    H_f, T_f = build_history(tx_f, fr, ref["prod"])
                    if sink_b is None:
                        probe = generate(W_f, fr.iloc[:500], H_f, T_f, tru, CFG.N_NEG_PER_CLIENT, rng_b)
                        rate = len(probe) / 500.0
                        del probe
                        gc.collect()
                        sink_b = MatrixSink(n_ex, rate * 1.15 + 2, FEATURES)
                    with timer(f"bag{bag} fold{f_idx} snap{snap} ({len(cc):,} exemples)"):
                        generate(W_f, fr, H_f, T_f, tru, CFG.N_NEG_PER_CLIENT, rng_b,
                                 desc=f"b{bag}f{f_idx}s{snap}", sink=sink_b,
                                 gid_offset=snap * (1 << 21))
                    del H_f, T_f
                    gc.collect()
                if f_idx == 0:
                    W_b, tx_b = W_f, tx_f
                else:
                    del W_f, tx_f
                    gc.collect()
            H_b, T_b = build_history(tx_b, pd.concat([fr_va, fr_es]), ref["prod"])
            D_tr_b = sink_b.done()
            D_es_b = generate(W_b, fr_es, H_b, T_b, truth, CFG.N_NEG_PER_CLIENT, rng_b)
            log(f"   {D_tr_b[0].shape[0]:,} lignes | {len(FEATURES)} features")
            m_b = train_ranker(D_tr_b, D_es_b, FEATURES, CFG.LGB1, CFG.ROUNDS1,
                               CFG.N_SEEDS1, f"bag{bag}", mdl_dir)
            del D_tr_b, D_es_b, sink_b
            gc.collect()
            # --- validation de ce bag, pendant que son monde est en vie ------
            parts = []
            for i in range(0, len(fr_va), CFG.CHUNK_CLIENTS):
                ch = generate(W_b, fr_va.iloc[i:i + CFG.CHUNK_CLIENTS], H_b, T_b, truth)
                ch["s"] = predict_rank(m_b, ch, FEATURES)
                ch["rk"] = ch.groupby("c")["s"].rank(ascending=False, method="first").astype("int16")
                if bag == 0:
                    cand_hits.append(ch.groupby("c")["y"].max())
                parts.append(ch[ch["rk"] <= CFG.K_FUSE][["c", "p", "y", "rk"]].copy())
                del ch
                gc.collect()
            V_b = pd.concat(parts, ignore_index=True)
            hb, _ = recall_at_k(top_k(V_b, -V_b["rk"].values.astype("float32")), truth)
            log(f"   BAG {bag + 1} seul : HitRate@5 = {hb:.4f}")
            bag_scores.append(hb)
            bag_tops.append(V_b)
            bag_models.append(m_b)
            del W_b, tx_b, H_b, T_b
            gc.collect()

        # --- fusion Borda des bags ------------------------------------------
        def borda(tops, k_out=5):
            F = None
            for i, V in enumerate(tops):
                v = V[["c", "p", "rk"]].rename(columns={"rk": f"rk{i}"})
                F = v if F is None else F.merge(v, on=["c", "p"], how="outer")
            F["b"] = 0.0
            for i in range(len(tops)):
                F["b"] += 1.0 / (F[f"rk{i}"].fillna(2 * CFG.K_FUSE).values + 3.0)
            F = F.sort_values(["c", "b"], ascending=[True, False], kind="mergesort")
            F["r"] = F.groupby("c").cumcount()
            return F[F["r"] < k_out]
        Fv = borda(bag_tops)
        top5v = {int(a_): list(b_) for a_, b_ in Fv.groupby("c")["p"].apply(list).items()}
        best_r, best_h = recall_at_k(top5v, truth)
        cand_recall = float(pd.concat(cand_hits).mean()) if cand_hits else float("nan")
        log("=" * 74)
        log(f"CANDIDATE RECALL (bag 0) = {cand_recall:.4f}")
        log("bags seuls : " + "  ".join(f"{x:.4f}" for x in bag_scores))
        log(f"FUSION DES {CFG.N_BAGS} BAGS : HitRate@5 = {best_r:.4f}  (Recall@5 = {best_h:.4f})  "
            f"n={len(va_ids):,} clients")
        log("=" * 74)
        r1, h1, r2, h2 = max(bag_scores), best_h, best_r, best_h
        best_w, best_lam = 0.0, 0.0
        curve = {}
        Ks = []
        models1 = [m for mb in bag_models for m in mb]
        models2 = []
        FEATURES2 = list(FEATURES)
        imp = pd.DataFrame({"f": FEATURES, "gain": np.mean(
            [m.feature_importance("gain") for m in models1], axis=0)})
        imp = imp.sort_values("gain", ascending=False)
        imp.to_csv(os.path.join(CFG.OUT_DIR, "feature_importance.csv"), index=False)
        log("Top 15 features (moyenne des bags) :\n" + imp.head(15).to_string(index=False))
        tr2_ids = np.zeros(0, dtype="int32")

        # --- INFERENCE : un seul monde complet, tous les bags le scorent -----
        with timer("monde INFER (toutes les transactions)"):
            W_in = World(tx, ref, "INFER")
        fr_te = frame_of(test_c, use_last=True)
        fr_te["t0"] = fr_te["t0"].fillna(int(tx["day"].max())).astype("int32")
        with timer("TEST - candidats + features"):
            H_te, T_te = build_history(tx, fr_te, ref["prod"])
            D_te = generate(W_in, fr_te, H_te, T_te)
        log(f"   {len(D_te):,} lignes ({len(D_te) / max(len(fr_te), 1):.0f} candidats/client)")
        tops_te = []
        for i, m_b in enumerate(bag_models):
            sc_b = predict_rank(m_b, D_te, FEATURES)
            V = D_te[["c", "p"]].copy()
            V["rk"] = pd.Series(sc_b).groupby(D_te["c"].values).rank(ascending=False, method="first").values.astype("int16")
            tops_te.append(V[V["rk"] <= CFG.K_FUSE])
        Ft = borda(tops_te)
        top5 = {int(a_): list(b_) for a_, b_ in Ft.groupby("c")["p"].apply(list).items()}
    else:
        n_examples = sum(len(snap_targets(fd, sn)[0]) for fd in folds for sn in range(CFG.N_SNAPSHOTS))
        log(f"exemples d'entrainement : {n_examples:,} "
            f"({len(tr_ids):,} clients x jusqu'a {CFG.N_SNAPSHOTS} instantanes, {len(folds)} folds)")

        models1 = []
        reload1 = args.no_train and os.path.exists(os.path.join(mdl_dir, "s1_0.txt"))
        sink = None
        W_tr = tx_tr = None
        for f_idx in reversed(range(len(folds))):        # fold 0 en dernier : son monde reste en vie
            fold = folds[f_idx]
            tx_f = world_for(fold, f"TRAIN-fold{f_idx}")
            W_f = World(tx_f, ref, f"TRAIN-fold{f_idx}")
            if not reload1:
                for snap in range(CFG.N_SNAPSHOTS):
                    cc, td, t0s = snap_targets(fold, snap)
                    if len(cc) == 0:
                        continue
                    fr = frame_at(cc, t0s)
                    tru = truth_of(cc, td)
                    with timer(f"historiques fold{f_idx}/snap{snap}"):
                        H_f, T_f = build_history(tx_f, fr, ref["prod"])
                    if sink is None:
                        probe = generate(W_f, fr.iloc[:500], H_f, T_f, tru, CFG.N_NEG_PER_CLIENT, rng)
                        rate = len(probe) / 500.0
                        del probe
                        gc.collect()
                        log(f"   sonde : {rate:.1f} lignes/exemple -> matrice {n_examples:,} x {rate * 1.15:.1f}")
                        sink = MatrixSink(n_examples, rate * 1.15 + 2, FEATURES)
                    with timer(f"ETAGE 2 - fold{f_idx}/snap{snap} ({len(cc):,} exemples)"):
                        generate(W_f, fr, H_f, T_f, tru, CFG.N_NEG_PER_CLIENT, rng,
                                 desc=f"f{f_idx}s{snap}", sink=sink, gid_offset=snap * (1 << 21))
                    del H_f, T_f
                    gc.collect()
            if f_idx == 0:
                W_tr, tx_tr = W_f, tx_f
            else:
                del W_f, tx_f
                gc.collect()

        with timer("historiques evaluation"):
            H_tr, T_tr = build_history(tx_tr, pd.concat([fr_va, fr_es]), ref["prod"])

        if reload1:
            for i in range(20):
                f_ = os.path.join(mdl_dir, f"s1_{i}.txt")
                if not os.path.exists(f_):
                    break
                models1.append(lgb.Booster(model_file=f_))
            with timer("features (reconstruction du schema)"):
                D_tmp = generate(W_tr, fr_es.iloc[:600], H_tr, T_tr, truth, CFG.N_NEG_PER_CLIENT, rng)
                del D_tmp
            log(f"{len(models1)} modele(s) etage 2 recharges")
        else:
            D_tr1 = sink.done()
            log(f"   {D_tr1[0].shape[0]:,} lignes | {len(FEATURES)} features")
            with timer("ETAGE 2 - jeu d'early stopping"):
                D_es1 = generate(W_tr, fr_es, H_tr, T_tr, truth, CFG.N_NEG_PER_CLIENT, rng)
            models1 = train_ranker(D_tr1, D_es1, FEATURES, CFG.LGB1, CFG.ROUNDS1,
                                   CFG.N_SEEDS1, "s1", mdl_dir)
            del D_tr1, D_es1, sink
            gc.collect()

        # l'etage 3 travaille sur le fold 0 (son monde est celui qui reste en vie)
        tr2_c, tr2_td, tr2_t0 = snap_targets(folds[0][:min(CFG.N_TRAIN2_CLIENTS, len(folds[0]))], 0)
        fr_tr2 = frame_at(tr2_c, tr2_t0)
        truth_tr2 = truth_of(tr2_c, tr2_td)
        tr2_ids = tr2_c

        def frame_of(cids, use_last=False):
            d = sp.reindex(cids)
            t0 = (d["last_day"] if use_last else d["t0"].fillna(d["last_day"])).values.astype("int32")
            return frame_at(cids, t0)

        # =====================================================================
        # DIAGNOSTIC + ELAGAGE sur la validation
        # =====================================================================
        Ks = [5, 10, 20, 50, 100, CFG.PRUNE_TO]
        diag = {"Ks": Ks, "cand": []}
        for K in Ks:
            diag[f"r{K}"] = []
        with timer("VALID - candidats, score etage 2, elagage"):
            D_va = generate_rank_prune(W_tr, fr_va, H_tr, T_tr, models1, CFG.PRUNE_TO,
                                       truth, diag, desc="valid")
        cand_recall = float(np.concatenate(diag["cand"]).mean())
        curve = {K: float(np.concatenate(diag[f"r{K}"]).mean()) for K in Ks}
        log("=" * 74)
        log(f"CANDIDATE RECALL (plafond du retrieval) = {cand_recall:.4f}")
        log("courbe de rappel du ranker 1 : " +
            "  ".join(f"@{K}={curve[K]:.4f}" for K in Ks))
        r1, h1 = recall_at_k(top_k(D_va, -D_va["s1_rank"].values), truth)
        log(f"ETAGE 2 seul              HitRate@5={r1:.4f}  (Recall@5={h1:.4f})")
        pd.DataFrame({"K": Ks, "recall": [curve[K] for K in Ks]}).to_csv(
            os.path.join(CFG.OUT_DIR, "recall_curve.csv"), index=False)

        # =====================================================================
        # ETAGE 3 : re-ranker sur les candidats survivants
        # =====================================================================
        # L'etage 3 n'utilise PAS le score de l'etage 2 comme feature : teste, il
        # capte 87 % du gain de l'arbre et le modele se contente de recopier
        # l'etage 2 (best_iter=12, score en baisse). Sans lui, l'etage 3 devient un
        # modele independant entraine sur des negatifs difficiles, et le melange
        # des deux gagne. Les colonnes restent presentes pour le melange final.
        FEATURES2 = list(FEATURES)
        models2 = []
        skip3 = CFG.N_TRAIN2_CLIENTS <= 0 or len(tr2_ids) == 0
        if skip3:
            log("ETAGE 3 desactive (--n-train2 0) : sortie sur l'etage 2 seul")
        elif args.no_train and os.path.exists(os.path.join(mdl_dir, "s2_0.txt")):
            for i in range(20):
                f_ = os.path.join(mdl_dir, f"s2_{i}.txt")
                if not os.path.exists(f_):
                    break
                models2.append(lgb.Booster(model_file=f_))
            log(f"{len(models2)} modele(s) etage 3 recharges")
        if not models2 and not skip3:
            with timer("ETAGE 3 - candidats elagues (entrainement)"):
                # les groupes sans positif sont ecartes a la volee (pos_only)
                sink2 = MatrixSink(len(fr_tr2), CFG.PRUNE_TO * (cand_recall * 1.12 + 0.05),
                                   FEATURES2)
                H_tr2, T_tr2 = build_history(tx_tr, fr_tr2, ref["prod"])
                D_tr2 = generate_rank_prune(W_tr, fr_tr2, H_tr2, T_tr2, models1, CFG.PRUNE_TO,
                                            truth_tr2, None, desc="etage3", sink=sink2, pos_only=True)
                del H_tr2, T_tr2
            log(f"   {D_tr2[0].shape[0]:,} lignes | {len(np.unique(D_tr2[2])):,} groupes avec positif")
            with timer("ETAGE 3 - jeu d'early stopping"):
                D_es2 = generate_rank_prune(W_tr, fr_es, H_tr, T_tr, models1, CFG.PRUNE_TO,
                                            truth, pos_only=True)
            models2 = train_ranker(D_tr2, D_es2, FEATURES2, CFG.LGB2,
                                   CFG.ROUNDS2, CFG.N_SEEDS2, "s2", mdl_dir)
            del D_tr2, D_es2, sink2
            gc.collect()

        # ---- score final sur la CV locale + arbitrage du melange ---------------
        #      w=1.00 (etage 2 seul) fait partie des candidats : le pipeline ne
        #      peut donc jamais finir en dessous de son etage 2.
        s1n = pd.Series(-D_va["s1_rank"].values).groupby(D_va["c"].values).rank(pct=True).values.astype("float32")
        if not models2:
            best_w, best_r, best_h, r2, h2 = 1.0, r1, h1, r1, h1
        else:
            s2 = predict_rank(models2, D_va, FEATURES2)
            r2, h2 = recall_at_k(top_k(D_va, s2), truth)
            log(f"ETAGE 3 (re-ranking)      HitRate@5={r2:.4f}  (Recall@5={h2:.4f})")
            best_w, best_r, best_h = 0.0, r2, h2
            for w in (0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00):
                rr, hh = recall_at_k(top_k(D_va, (1 - w) * s2 + w * s1n), truth)
                log(f"   melange etage2 w={w:.2f} -> HitRate@5={rr:.4f}")
                if rr > best_r + 1e-5:
                    best_w, best_r, best_h = w, rr, hh
        # --- diversification sous Hit Rate : on teste plusieurs intensites -------
        sc_best = s1n if best_w >= 1.0 else ((1 - best_w) * s2 + best_w * s1n if models2 else s1n)
        bsize = dict(zip(D_va["c"].values, D_va["cl_basket"].values))
        best_lam = 0.0
        for lam in (0.15, 0.30, 0.50):
            rr, _ = recall_at_k(top_k_diverse(D_va, sc_best, W_tr, lam, 5, bsize), truth)
            log(f"   diversification lambda={lam:.2f} -> HitRate@5={rr:.4f}")
            if rr > best_r + 1e-5:
                best_lam, best_r = lam, rr
        log(f"RETENU : w_etage2={best_w:.2f}  lambda_diversite={best_lam:.2f}")

        log("=" * 74)
        log(f"RESULTAT LOCAL : HitRate@5={best_r:.4f}  (Recall@5={best_h:.4f})  "
            f"w_etage2={best_w:.2f}  n={len(va_ids):,} clients")
        log("=" * 74)

        mref = models2[0] if models2 else models1[0]
        fref = FEATURES2 if models2 else FEATURES
        imp = pd.DataFrame({"f": fref, "gain": mref.feature_importance("gain")})
        imp = imp.sort_values("gain", ascending=False)
        imp.to_csv(os.path.join(CFG.OUT_DIR, "feature_importance.csv"), index=False)
        log("Top 15 features :\n" + imp.head(15).to_string(index=False))
        del D_va
        gc.collect()

        # =====================================================================
        # 9. INFERENCE : monde complet, clients de test
        # =====================================================================
        with timer("monde INFER (toutes les transactions)"):
            W_in = World(tx, ref, "INFER")
        fr_te = frame_of(test_c, use_last=True)
        fr_te["t0"] = fr_te["t0"].fillna(int(tx["day"].max())).astype("int32")
        with timer("TEST - candidats, etage 2, elagage"):
            H_te, T_te = build_history(tx, fr_te, ref["prod"])
            D_te = generate_rank_prune(W_in, fr_te, H_te, T_te, models1, CFG.PRUNE_TO)
        log(f"   {len(D_te):,} lignes ({len(D_te) / max(len(fr_te), 1):.0f} candidats retenus/client)")

        s1t = pd.Series(-D_te["s1_rank"].values).groupby(D_te["c"].values).rank(pct=True).values.astype("float32")
        sc = s1t if not models2 else predict_rank(models2, D_te, FEATURES2)
        if models2 and best_w > 0:
            sc = (1 - best_w) * sc + best_w * s1t
        bs_te = dict(zip(D_te["c"].values, D_te["cl_basket"].values))
        top5 = top_k_diverse(D_te, sc, W_in, best_lam, 5, bs_te) if best_lam > 0 else top_k(D_te, sc)

    # ---- filet de securite : aucun client ne sort avec moins de 5 produits --
    fb_cache = {}
    for c_, mi_ in zip(fr_te["c"].values, fr_te["mi0"].values):
        cur = top5.get(int(c_), [])
        if len(cur) >= 5:
            continue
        ci = int(max(ref["ccountry"][int(c_)], 0))
        if (int(mi_), ci) not in fb_cache:
            fb = list(W_in.top_pop(mi_, ci, 3, 40))
            if len(fb) < 15:
                fb += list(W_in.top_pop(mi_, 0, 3, 40, "G"))
            fb_cache[(int(mi_), ci)] = fb
        for q in fb_cache[(int(mi_), ci)]:
            if len(cur) >= 5:
                break
            if q not in cur:
                cur.append(int(q))
        top5[int(c_)] = cur

    # =====================================================================
    # 10. SOUMISSION - IDs restitues en STRING pure
    # =====================================================================
    pv = ref["pv"]

    def rows(codes, key):
        out = []
        for c_ in codes:
            it = top5.get(int(c_), [])[:5]
            it = it + [it[-1] if it else 0] * (5 - len(it))
            out.append([cv.keys[int(c_)]] + [pv.keys[int(x)] for x in it] + [key])
        return out

    sub = pd.DataFrame(rows(tv_c, "Validation") + rows(tf_c, "Final"),
                       columns=["ClientID", "item_1", "item_2", "item_3", "item_4", "item_5", "KEY"])
    idc = ["ClientID", "item_1", "item_2", "item_3", "item_4", "item_5"]
    for col in idc:
        sub[col] = sub[col].astype(str)
    assert len(sub) == len(tval) + len(tfin), "nombre de lignes incorrect"
    assert sub[idc].apply(lambda s: s.str.fullmatch(r"\d{8,20}")).all().all(), \
        "identifiant non numerique (notation scientifique ?)"
    assert sub["ClientID"].nunique() == len(sub), "ClientID duplique"
    log(f"controles OK | items distincts par ligne : "
        f"{sub[idc[1:]].nunique(axis=1).mean():.2f}")

    sub.to_csv(os.path.join(CFG.OUT_DIR, "submission_all.csv"), index=False)
    sub[sub.KEY == "Validation"].to_csv(os.path.join(CFG.OUT_DIR, "submission_validation.csv"), index=False)
    sub[sub.KEY == "Final"].to_csv(os.path.join(CFG.OUT_DIR, "submission_final.csv"), index=False)

    cfg_txt = (f"n_tr={len(tr_ids)};n_tr2={len(tr2_ids)};n_neg={CFG.N_NEG_PER_CLIENT};"
               f"prune={CFG.PRUNE_TO};seeds={CFG.N_SEEDS1}/{CFG.N_SEEDS2};"
               f"lr1={CFG.LGB1['learning_rate']};lr2={CFG.LGB2['learning_rate']};"
               f"kctry={CFG.K_POP_COUNTRY};kstore={CFG.K_STORE_LAST};w2={best_w};"
               f"lam={best_lam};folds={CFG.N_FOLDS};snaps={CFG.N_SNAPSHOTS};bags={CFG.N_BAGS}")
    mins = (time.time() - t_start) / 60
    with open(os.path.join(CFG.OUT_DIR, "run_report.txt"), "w") as fh:
        fh.write(f"hitrate@5_local={best_r:.4f}\nrecall@5_local={best_h:.4f}\n"
                 f"hitrate@5_etage2={r1:.4f}\nhitrate@5_etage3={r2:.4f}\n"
                 f"candidate_recall={cand_recall:.4f}\n"
                 + "".join(f"ranker1_recall@{K}={curve[K]:.4f}\n" for K in Ks)
                 + f"n_valid={len(va_ids)}\nfeatures={len(FEATURES2)}\nminutes={mins:.1f}\n"
                   f"config={cfg_txt}\n")
    jp = os.path.join(CFG.OUT_DIR, "journal.csv")
    new = not os.path.exists(jp)
    with open(jp, "a") as fh:
        if new:
            fh.write("timestamp,hitrate5_local,recall5_local,hitrate5_etage2,"
                     "candidate_hit,minutes,score_leaderboard,config\n")
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')},{best_r:.4f},{best_h:.4f},{r1:.4f},"
                 f"{cand_recall:.4f},{mins:.1f},,{cfg_txt}\n")
    log(f"Soumissions ecrites dans {CFG.OUT_DIR}")
    log("journal.csv mis a jour -- y reporter a la main le score du leaderboard")
    log("NE JAMAIS ouvrir ces CSV dans Excel (conversion en 6.63E+16)")
    log(f"Temps total : {mins:.1f} min")


if __name__ == "__main__":
    main()
