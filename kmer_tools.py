"""
Script permettant de compter, trier et comparer des centaines de millions de
k-mères en quelques minutes, en confiant ses boucles à numba et en répartissant
le travail sur tous les cœurs.

Travaille sur des k-mères encodés, rangés en tableaux de nombres ; ne revient
aux lettres qu'au dernier moment, pour décrire ses trouvailles.

Les fonctions les moins évidentes :
    find_snp_pairs          paires de k-mères différant de n substitutions
    is_in_sorted            appartenance de k-mères à une table triée
    count_kmers_batch       k-mères d'un lot de reads, comptés
    MergeTree               réunion en arbre de comptages arrivant un à un
    compile_kernels         mise en route des fonctions compilées

Un k-mère est l'entier de 2k bits décrit dans nucleotide_2bits, découpé en
LIMBES de 64 bits puisque au-delà de k=32 il ne tient plus dans un entier
machine. Un ensemble de k-mères est un tableau à n_limbs colonnes, une ligne
par k-mère, limbe de tête en colonne 0 : comparer deux lignes colonne par
colonne revient à comparer les deux entiers.

Contraintes : les tableaux échangés sont triés, et le rester conditionne leur
rapidité ; seules les primitives _limb_ connaissent le découpage en limbes ;
les fonctions njit sont lentes au premier appel, rapides ensuite.
"""
import time
from typing import List, Tuple

import numpy as np
from numba import njit, prange

import nucleotide_2bits as n2b

BITS_PER_LIMB = 64


# ---------------------------------------------------------------------------
# Découpage d'un k-mère en limbes
# ---------------------------------------------------------------------------

# Vérifie que la longueur k d'un k-mère est valide. Il n'y a pas de maximum :
# le nombre de limbes s'ajuste à k.
def check_k_supported(k: int) -> None:
    if k < 1:
        raise ValueError(f"k={k} invalide : k doit être >= 1.")


def n_limbs_for_k(k: int) -> int:
    """Renvoie le nombre de limbes qu'il faut pour porter un k-mère."""
    return (n2b.kmer_bits(k) + BITS_PER_LIMB - 1) // BITS_PER_LIMB


def top_limb_mask(k: int) -> np.uint64:
    """
    Renvoie le masque du limbe de tête, le seul à n'être pas entièrement
    occupé : les 2k bits d'un k-mère ne tombent presque jamais juste sur un
    multiple de 64. Appliquer ce masque efface les bits qui dépassent.
    """
    top_bits = n2b.kmer_bits(k) - BITS_PER_LIMB * (n_limbs_for_k(k) - 1)
    return np.uint64((1 << top_bits) - 1)


def empty_kmers(k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renvoie un comptage vide, aux dimensions attendues par la suite du
    traitement : aucun k-mère, aucune occurrence.
    """
    return np.empty((0, n_limbs_for_k(k)), dtype=np.uint64), np.empty(0, dtype=np.int64)


def limbs_to_int(row: np.ndarray) -> int:
    """Renvoie l'entier que forment les limbes d'un k-mère mis bout à bout."""
    value = 0
    for limb in row.tolist():
        value = (value << BITS_PER_LIMB) | limb
    return value


def decode_kmer_row(row: np.ndarray, k: int) -> str:
    """Traduit une ligne de limbes en la chaîne de lettres qu'elle porte."""
    return n2b.decode_kmer(limbs_to_int(row), k)


def encode_kmer_row(sequence: str, k: int) -> np.ndarray:
    """Traduit une chaîne de k lettres en la ligne de limbes qui la représente."""
    value = 0
    for letter in sequence:
        value = (value << n2b.NUCLEOTIDE_BITS) | n2b.NUCLEOTID_MAP[ord(letter)]
    row = np.empty(n_limbs_for_k(k), dtype=np.uint64)
    for limb in range(row.shape[0] - 1, -1, -1):
        row[limb] = np.uint64(value & ((1 << BITS_PER_LIMB) - 1))
        value >>= BITS_PER_LIMB
    return row


@njit(cache=True, inline="always")
def _limb_slot(n_limbs: int, k: int, pos: int):
    """
    Renvoie où se trouve le nucléotide de rang pos : le limbe qui le porte, et
    son décalage à l'intérieur de ce limbe.

    Un nucléotide ne chevauche jamais deux limbes, car la taille d'un limbe
    est un multiple de deux bits.
    """
    shift = n2b.shift_of(k, pos)
    return n_limbs - 1 - shift // BITS_PER_LIMB, np.uint64(shift % BITS_PER_LIMB)


@njit(cache=True, inline="always")
def _limb_read(row: np.ndarray, limb: int, shift: np.uint64) -> np.int64:
    """Renvoie le code du nucléotide logé dans ce limbe, à ce décalage."""
    return np.int64((row[limb] >> shift) & np.uint64(n2b.NUCLEOTIDE_MASK))


@njit(cache=True, inline="always")
def _limb_write(row: np.ndarray, limb: int, shift: np.uint64, code: int) -> None:
    """Remplace le nucléotide logé dans ce limbe, à ce décalage, par un autre."""
    cleared = row[limb] & ~(np.uint64(n2b.NUCLEOTIDE_MASK) << shift)
    row[limb] = cleared | (np.uint64(code) << shift)


@njit(cache=True, inline="always")
def _limb_push(row: np.ndarray, code: int, top_mask: np.uint64) -> None:
    """
    Ajoute un nucléotide à la droite du k-mère et écarte celui de gauche, ce
    qui fait avancer la fenêtre glissante d'un cran.

    C'est l'opération sur laquelle repose tout l'encodage : sans elle, il
    faudrait reconstruire le k-mère entier à chaque position d'un read.
    """
    carry = np.uint64(code)
    keep = np.uint64(BITS_PER_LIMB - n2b.NUCLEOTIDE_BITS)
    for limb in range(row.shape[0] - 1, -1, -1):
        overflow = row[limb] >> keep
        row[limb] = (row[limb] << np.uint64(n2b.NUCLEOTIDE_BITS)) | carry
        carry = overflow
    row[0] &= top_mask


# ---------------------------------------------------------------------------
# Comparaison et recherche (remplacent np.searchsorted / np.isin)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _compare_rows(a: np.ndarray, b: np.ndarray) -> int:
    """
    Compare deux k-mères et renvoie -1 si le premier est le plus petit, 0
    s'ils sont égaux, et 1 sinon.
    """
    for t in range(a.shape[0]):
        if a[t] < b[t]:
            return -1
        if a[t] > b[t]:
            return 1
    return 0


@njit(cache=True)
def _find_row(sorted_kmers: np.ndarray, query: np.ndarray) -> int:
    """
    Cherche un k-mère dans un tableau trié et renvoie son rang, ou -1 s'il n'y
    figure pas.
    """
    lo = 0
    hi = sorted_kmers.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        c = _compare_rows(sorted_kmers[mid], query)
        if c < 0:
            lo = mid + 1
        elif c > 0:
            hi = mid
        else:
            return mid
    return -1


@njit(cache=True)
def is_in_sorted(queries: np.ndarray, sorted_kmers: np.ndarray) -> np.ndarray:
    """
    Prend deux tableaux triés et dit, pour chaque k-mère du premier, s'il
    figure dans le second.

    Les deux étant triés, un seul parcours simultané suffit, là où une
    recherche binaire par k-mère interrogé coûterait bien plus cher.
    """
    n = queries.shape[0]
    m = sorted_kmers.shape[0]
    found = np.zeros(n, dtype=np.bool_)
    j = 0
    for i in range(n):
        while j < m and _compare_rows(sorted_kmers[j], queries[i]) < 0:
            j += 1
        if j == m:
            break
        found[i] = _compare_rows(sorted_kmers[j], queries[i]) == 0
    return found


# ---------------------------------------------------------------------------
# Tri des k-mères
# ---------------------------------------------------------------------------
#
# Le tri d'une seule colonne de nombres est nettement plus rapide que celui de
# plusieurs, d'où un chemin dédié au cas à un limbe.

def _sorted_order(kmers: np.ndarray) -> np.ndarray:
    """Renvoie l'ordre dans lequel lire les k-mères pour les obtenir triés."""
    # Colonnes recopiées de façon contiguë : np.lexsort est sensible à leur
    # disposition en mémoire. Elles sont données du limbe le plus faible au
    # plus fort, car sa dernière clé est la plus prioritaire.
    columns = [np.ascontiguousarray(kmers[:, j]) for j in range(kmers.shape[1] - 1, -1, -1)]
    return np.lexsort(tuple(columns))


def _group_starts(sorted_kmers: np.ndarray) -> np.ndarray:
    """
    Renvoie, dans un tableau trié, le rang où commence chaque k-mère distinct.
    """
    is_new = np.empty(sorted_kmers.shape[0], dtype=bool)
    is_new[0] = True
    is_new[1:] = np.any(sorted_kmers[1:] != sorted_kmers[:-1], axis=1)
    return np.flatnonzero(is_new)


def sort_unique_kmers(kmers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trie les k-mères, réunit les exemplaires identiques et renvoie les k-mères
    distincts avec le nombre d'exemplaires de chacun.
    """
    n, n_limbs = kmers.shape
    if n == 0:
        return kmers, np.empty(0, dtype=np.int64)
    if n_limbs == 1:
        unique, counts = np.unique(kmers[:, 0], return_counts=True)
        return unique.reshape(-1, 1), counts.astype(np.int64)

    sorted_kmers = kmers[_sorted_order(kmers)]
    starts = _group_starts(sorted_kmers)
    return sorted_kmers[starts], np.diff(np.append(starts, n)).astype(np.int64)


# ---------------------------------------------------------------------------
# Comptage des k-mères : kernel + unité de travail
# ---------------------------------------------------------------------------

@njit(cache=True)
def _encode_batch_kmers(
    codes: np.ndarray, read_ends: np.ndarray, k: int, top_mask: np.uint64, out: np.ndarray
) -> int:
    """
    Encode tous les k-mères d'un lot de reads mis bout à bout, read_ends
    donnant la position où chaque read s'arrête. Écrit les k-mères dans out et
    renvoie combien il en a écrit.

    Un nucléotide invalide interrompt la fenêtre glissante, qui repart du
    nucléotide suivant : aucun k-mère ne l'enjambe.
    """
    current = np.zeros(out.shape[1], dtype=np.uint64)
    count = 0
    start = 0
    for r in range(read_ends.shape[0]):
        end = read_ends[r]
        current[:] = 0
        valid_len = 0
        for i in range(start, end):
            c = codes[i]
            if c == n2b.INVALID_CODE:  # base invalide : on recommence la fenêtre
                current[:] = 0
                valid_len = 0
                continue
            _limb_push(current, c, top_mask)
            valid_len += 1
            if valid_len >= k:
                out[count, :] = current
                count += 1
        start = end
    return count


def count_kmers_batch(seq_list: List[bytes], k: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Encode un lot de reads, replie chaque k-mère sur son complément inverse,
    et renvoie les k-mères distincts triés, leurs occurrences dans le lot, et
    le temps passé à chacune des trois phases.

    Le repliage a lieu ici, avant le tri du lot, plutôt que sur la table
    entière une fois tout compté : le tri qui suit devait avoir lieu de toute
    façon, et les lots sont traités en parallèle alors qu'une table complète
    se retrierait sur un seul cœur.

    Les temps sont mesurés ici parce que l'appelant fait travailler plusieurs
    lots en même temps et ne voit d'eux qu'une attente globale.
    """
    lengths = np.fromiter((len(s) for s in seq_list), dtype=np.int64, count=len(seq_list))
    # Un read de longueur L donne L-k+1 k-mères, 0 s'il est plus court que k.
    n_max = int(np.maximum(lengths - k + 1, 0).sum())
    if n_max == 0:
        kmers, counts = empty_kmers(k)
        return kmers, counts, {"encodage": 0.0, "repliage": 0.0, "tri": 0.0}

    t0 = time.perf_counter()
    codes = n2b.CODE_BY_BYTE[np.frombuffer(b"".join(seq_list), dtype=np.uint8)]
    read_ends = np.cumsum(lengths)
    out = np.empty((n_max, n_limbs_for_k(k)), dtype=np.uint64)
    n_written = _encode_batch_kmers(codes, read_ends, k, top_limb_mask(k), out)
    t_encode = time.perf_counter() - t0

    # Repliage sur place : les deux brins donnent deux écritures d'un même
    # endroit du génome, on n'en garde qu'une.
    t0 = time.perf_counter()
    _canonicalise_rows(out[:n_written], k, out[:n_written])
    t_fold = time.perf_counter() - t0

    t0 = time.perf_counter()
    kmers, counts = sort_unique_kmers(out[:n_written])
    return kmers, counts, {"encodage": t_encode, "repliage": t_fold,
                           "tri": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Repliage sur le complément inverse (appelé par snpDetect.py après le comptage)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _reverse_complement_row(row: np.ndarray, k: int, out: np.ndarray) -> None:
    """
    Écrit dans out le complément inverse d'un k-mère : chaque nucléotide
    devient son partenaire et passe du rang pos au rang k-1-pos.
    """
    n_limbs = row.shape[0]
    for j in range(n_limbs):
        out[j] = np.uint64(0)
    for pos in range(k):
        limb, shift = _limb_slot(n_limbs, k, pos)
        mirror_limb, mirror_shift = _limb_slot(n_limbs, k, k - 1 - pos)
        code = _limb_read(row, limb, shift)
        _limb_write(out, mirror_limb, mirror_shift, n2b.complement_code(code))


@njit(cache=True, inline="always")
def _canonical_row(row: np.ndarray, k: int, mirror: np.ndarray) -> np.ndarray:
    """
    Renvoie la forme canonique d'un k-mère, c'est-à-dire la plus petite de ses
    deux écritures, la sienne et celle de l'autre brin. mirror sert de
    brouillon.

    Un read venant indifféremment de l'un ou l'autre brin, deux écritures du
    même k-mère circulent ; n'en retenir qu'une évite de les compter à part.
    """
    _reverse_complement_row(row, k, mirror)
    if _compare_rows(row, mirror) <= 0:
        return row
    return mirror


@njit(cache=True)
def find_kmer(row: np.ndarray, sorted_kmers: np.ndarray, k: int) -> int:
    """
    Renvoie le rang d'un k-mère dans une table repliée et triée, ou -1 s'il
    n'y figure pas.
    """
    mirror = np.empty(row.shape[0], dtype=np.uint64)
    return _find_row(sorted_kmers, _canonical_row(row, k, mirror))


@njit(cache=True)
def _canonicalise_rows(kmers: np.ndarray, k: int, out: np.ndarray) -> None:
    """
    Écrit dans out la forme canonique de chaque k-mère du tableau. out peut
    être le tableau lui-même, aucune ligne n'en regardant une autre.

    La boucle est volontairement séquentielle : elle tourne déjà dans l'un des
    workers, et y ajouter des fils ne ferait que les mettre en concurrence.
    """
    n_limbs = kmers.shape[1]
    mirror = np.empty(n_limbs, dtype=np.uint64)
    for i in range(kmers.shape[0]):
        out[i, :] = _canonical_row(kmers[i], k, mirror)


def sort_and_sum_counts(kmers: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trie des k-mères qui portent déjà des occurrences et additionne celles des
    k-mères identiques. Renvoie les k-mères distincts et leurs occurrences
    cumulées.
    """
    n, n_limbs = kmers.shape
    if n == 0:
        return kmers, counts
    order = np.argsort(kmers[:, 0], kind="stable") if n_limbs == 1 else _sorted_order(kmers)
    sorted_kmers, sorted_counts = kmers[order], counts[order]
    starts = _group_starts(sorted_kmers)
    return sorted_kmers[starts], np.add.reduceat(sorted_counts, starts)


# ---------------------------------------------------------------------------
# Fusion de deux comptages (appelée par snpDetect.py à chaque lot qui arrive)
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _merge_sorted_unique_kmers(kmers_a, counts_a, kmers_b, counts_b):
    """
    Fusionne deux comptages triés et sans doublon en un seul, en additionnant
    les occurrences des k-mères communs.

    Une fois compilée, cette fonction se passe de l'interpréteur Python : elle
    peut donc travailler dans un fil séparé pendant que le programme continue
    de lire le fichier.
    """
    n = kmers_a.shape[0]
    m = kmers_b.shape[0]
    out = np.empty((n + m, kmers_a.shape[1]), dtype=np.uint64)
    out_counts = np.empty(n + m, dtype=np.int64)
    i = 0
    j = 0
    out_idx = 0
    while i < n and j < m:
        c = _compare_rows(kmers_a[i], kmers_b[j])
        if c == 0:
            out[out_idx, :] = kmers_a[i]
            out_counts[out_idx] = counts_a[i] + counts_b[j]
            i += 1
            j += 1
        elif c < 0:
            out[out_idx, :] = kmers_a[i]
            out_counts[out_idx] = counts_a[i]
            i += 1
        else:
            out[out_idx, :] = kmers_b[j]
            out_counts[out_idx] = counts_b[j]
            j += 1
        out_idx += 1
    while i < n:
        out[out_idx, :] = kmers_a[i]
        out_counts[out_idx] = counts_a[i]
        i += 1
        out_idx += 1
    while j < m:
        out[out_idx, :] = kmers_b[j]
        out_counts[out_idx] = counts_b[j]
        j += 1
        out_idx += 1
    return out[:out_idx], out_counts[:out_idx]


def merge_kmer_counts(
    kmers_a: np.ndarray, counts_a: np.ndarray, kmers_b: np.ndarray, counts_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fusionne deux comptages triés en un seul, en additionnant les occurrences
    des k-mères communs, et renvoie l'autre tel quel si l'un des deux est
    vide.
    """
    if kmers_a.shape[0] == 0:
        return kmers_b, counts_b
    if kmers_b.shape[0] == 0:
        return kmers_a, counts_a
    return _merge_sorted_unique_kmers(kmers_a, counts_a, kmers_b, counts_b)


class MergeTree:
    """
    Réunit des comptages qui arrivent un à un, en fusionnant d'abord les plus
    petits entre eux plutôt que chacun dans un total qui grossit.

    Versé dans un total commun, un comptage est recopié par toutes les fusions
    qui suivent la sienne, soit la moitié du nombre de lots en moyenne.
    Fusionné par paires de taille voisine, il ne l'est qu'une fois par niveau
    de l'arbre, soit le logarithme de ce nombre. Le pic de mémoire, lui, ne
    change pas : la dernière fusion réunit toujours deux moitiés en un tout.
    """

    def __init__(self, k: int):
        self.k = k
        self.levels: List[Tuple[int, np.ndarray, np.ndarray]] = []

    def add(self, kmers: np.ndarray, counts: np.ndarray) -> None:
        """
        Ajoute un comptage à l'arbre, et fusionne aussitôt les tables de même
        rang que son arrivée fait se rencontrer.
        """
        rank = 1
        while self.levels and self.levels[-1][0] == rank:
            _, waiting_kmers, waiting_counts = self.levels.pop()
            kmers, counts = merge_kmer_counts(waiting_kmers, waiting_counts, kmers, counts)
            rank += 1
        self.levels.append((rank, kmers, counts))

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fusionne les tables qui restent, de la plus petite à la plus grande, et
        renvoie le comptage unique qu'elles forment.
        """
        kmers, counts = empty_kmers(self.k)
        while self.levels:
            _, waiting_kmers, waiting_counts = self.levels.pop()
            kmers, counts = merge_kmer_counts(kmers, counts, waiting_kmers, waiting_counts)
        return kmers, counts


# ---------------------------------------------------------------------------
# Recherche de SNPs
# ---------------------------------------------------------------------------
#
# Pour un candidat, on énumère les k-mères qui en diffèrent d'exactement m
# substitutions et on cherche lesquels existent dans l'autre souche. Leur
# nombre croît en k^m, ce qui borne en pratique les valeurs de m utilisables.


@njit(cache=True)
def _apply_substitutions(mutant, k, positions, alt_ranks, saved_limbs):
    """
    Remplace, à chacune des positions données, le nucléotide de mutant par
    l'un des trois autres, désigné par le rang correspondant dans alt_ranks.

    Renvoie les nucléotides remplacés et note dans saved_limbs les limbes
    touchés, ce qui permet à l'appelant de revenir en arrière sans recopier le
    k-mère entier.
    """
    n_limbs = mutant.shape[0]
    originals = np.empty(positions.shape[0], dtype=np.uint8)
    for i in range(positions.shape[0]):
        limb, shift = _limb_slot(n_limbs, k, positions[i])
        saved_limbs[i] = limb
        current = _limb_read(mutant, limb, shift)
        originals[i] = np.uint8(current)
        _limb_write(mutant, limb, shift, n2b.alternative_code(current, alt_ranks[i]))
    return originals


@njit(cache=True)
def _scan_candidate(candidates, c, ref_kmers, k, m, write, start,
                    ref_idx, out_positions, out_orig, out_alt):
    """
    Compte les k-mères de ref_kmers qui diffèrent du candidat c par exactement
    m substitutions, et les enregistre quand write vaut True.

    Toutes les façons de choisir m positions parmi k sont essayées, et pour
    chacune les trois autres nucléotides à chaque position : le travail croît
    donc comme k puissance m, ce qui borne en pratique les valeurs de m.
    """
    n_limbs = candidates.shape[1]
    mutant = np.empty(n_limbs, dtype=np.uint64)
    mirror = np.empty(n_limbs, dtype=np.uint64)
    positions = np.empty(m, dtype=np.int64)
    alt_ranks = np.empty(m, dtype=np.int64)
    saved_limbs = np.empty(m, dtype=np.int64)
    for i in range(m):
        positions[i] = i

    n_combinations = 1
    for _ in range(m):
        n_combinations *= 3

    n_hits = 0
    while True:
        for code in range(n_combinations):
            rest = code
            for i in range(m):
                alt_ranks[i] = rest % 3
                rest //= 3
            mutant[:] = candidates[c]
            originals = _apply_substitutions(mutant, k, positions, alt_ranks, saved_limbs)

            # Le mutant est cherché sous sa forme canonique.
            j = _find_row(ref_kmers, _canonical_row(mutant, k, mirror))

            if j >= 0:
                if write:
                    o = start + n_hits
                    ref_idx[o] = j
                    for i in range(m):
                        out_positions[o, i] = positions[i]
                        out_orig[o, i] = originals[i]
                        limb, shift = _limb_slot(n_limbs, k, positions[i])
                        out_alt[o, i] = np.uint8(_limb_read(mutant, limb, shift))
                n_hits += 1

        # jeu de positions suivant
        i = m - 1
        while i >= 0 and positions[i] == k - m + i:
            i -= 1
        if i < 0:
            break
        positions[i] += 1
        for j2 in range(i + 1, m):
            positions[j2] = positions[j2 - 1] + 1
    return n_hits


@njit(cache=True, parallel=True)
def _find_snp_pairs_kernel(candidates: np.ndarray, ref_kmers: np.ndarray, k: int, m: int):
    """
    Cherche, pour tous les candidats à la fois, les k-mères de ref_kmers qui
    en diffèrent par m substitutions, et renvoie les trouvailles sous forme de
    tableaux.

    La recherche est faite deux fois, d'abord pour compter puis pour écrire.
    Chaque cœur connaît ainsi d'avance la zone du résultat qui lui revient et
    y écrit sans jamais attendre les autres.
    """
    n_c = candidates.shape[0]
    hits = np.zeros(n_c, dtype=np.int64)
    no_i = np.empty(0, dtype=np.int64)
    no_i2 = np.empty((0, m), dtype=np.int64)
    no_u2 = np.empty((0, m), dtype=np.uint8)
    for c in prange(n_c):
        hits[c] = _scan_candidate(candidates, c, ref_kmers, k, m, False, 0, no_i, no_i2, no_u2, no_u2)

    offsets = np.zeros(n_c + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(hits)
    total = offsets[n_c]
    cand_idx = np.empty(total, dtype=np.int64)
    ref_idx = np.empty(total, dtype=np.int64)
    out_positions = np.empty((total, m), dtype=np.int64)
    out_orig = np.empty((total, m), dtype=np.uint8)
    out_alt = np.empty((total, m), dtype=np.uint8)
    for c in prange(n_c):
        if hits[c] > 0:
            _scan_candidate(candidates, c, ref_kmers, k, m, True, offsets[c],
                            ref_idx, out_positions, out_orig, out_alt)
            for t in range(offsets[c], offsets[c + 1]):
                cand_idx[t] = c
    return cand_idx, ref_idx, out_positions, out_orig, out_alt


def find_snp_pairs(
    candidates: np.ndarray, candidate_counts: np.ndarray,
    ref_kmers: np.ndarray, ref_counts: np.ndarray, k: int, n_substitutions: int = 1,
) -> Tuple[List[dict], np.ndarray]:
    """
    Cherche, pour chaque candidat, les k-mères de ref_kmers qui en diffèrent
    par exactement n_substitutions nucléotides. Chaque trouvaille est une
    paire.

    Renvoie les paires, décrites cette fois en lettres, et la liste des
    candidats pour lesquels au moins une paire a été trouvée.
    """
    explained = np.zeros(candidates.shape[0], dtype=bool)
    if candidates.shape[0] == 0 or ref_kmers.shape[0] == 0:
        return [], explained

    cand_idx, ref_idx, positions, origs, alts = _find_snp_pairs_kernel(
        candidates, ref_kmers, k, n_substitutions)
    explained[cand_idx] = True

    # Reconstruire kmer_dna2 à partir de kmer_dna1, plutôt que le décoder
    # depuis la table, garantit que les deux k-mères d'une paire sont écrits
    # dans la même orientation.
    pairs = []
    for row in range(cand_idx.shape[0]):
        kmer_dna1 = decode_kmer_row(candidates[cand_idx[row]], k)
        letters = list(kmer_dna1)
        for i in range(n_substitutions):
            letters[positions[row, i]] = n2b.decode_nucleotide(int(alts[row, i]))
        pairs.append({
            "kmer_dna1": kmer_dna1,
            "kmer_dna2": "".join(letters),
            "count_dna1": int(candidate_counts[cand_idx[row]]),
            "count_dna2": int(ref_counts[ref_idx[row]]),
            "positions": tuple(int(p) for p in positions[row]),
            "alleles_dna1": "".join(n2b.decode_nucleotide(int(x)) for x in origs[row]),
            "alleles_dna2": "".join(n2b.decode_nucleotide(int(x)) for x in alts[row]),
        })
    return pairs, explained


# ---------------------------------------------------------------------------
# Mise en route des fonctions compilées
# ---------------------------------------------------------------------------

def compile_kernels(k: int) -> None:
    """
    Appelle une fois chaque fonction compilée, sur des données minuscules, pour
    que Numba les traduise en code machine avant le vrai travail.

    Sans cette mise en route, la traduction a lieu au premier appel, c'est-à-dire
    dans chacun des workers à la fois : son coût, qui se compte en dizaines de
    secondes, se mêle alors à celui du premier fichier compté. Faite ici, elle a
    lieu une fois, et les workers lisent le résultat dans le cache sur disque.
    """
    read = b"ACGTTGCA" * k
    kmers, counts, _ = count_kmers_batch([read], k)
    merge_kmer_counts(kmers, counts, kmers, counts)
    sort_and_sum_counts(kmers, counts)
    is_in_sorted(kmers, kmers)
    find_kmer(kmers[0], kmers, k)
    find_snp_pairs(kmers, counts, kmers, counts, k, n_substitutions=1)
