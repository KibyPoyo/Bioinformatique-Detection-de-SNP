"""
Script permettant de trouver les positions où deux souches d'une même espèce
portent un nucléotide différent, à partir des seuls fichiers de reads : aucun
génome de référence n'est utilisé.

    python snpDetect.py readsA.fasta readsB.fasta [--k 31] [--bruit 1]

C'est le fichier à lancer, et le seul qui décide de l'enchaînement : il ne
calcule presque rien, il appelle et il affiche.

    0. compiler les fonctions Numba              compile_kernels
    1. compter les k-mères de chaque souche      get_count_kmer_in_genome
    2. écarter ceux nés d'erreurs de séquençage  clean_kmer_with_errors
    3. comparer les deux souches                 find_snp_between_clean_kmers
    4. afficher les SNPs et le diagnostic        print_snp_list

Un SNP fait apparaître, dans une souche, des k-mères absents de l'autre qui ne
diffèrent de leurs jumeaux que par le nucléotide muté. Tout le reste consiste à
rendre cette comparaison praticable sur des centaines de millions de mots.

Contraintes : les données attendues ne portent que des substitutions ; la
recherche s'arrête à MAX_SUBSTITUTIONS par k-mère, son coût croissant comme une
puissance de ce nombre ; --bruit donne le taux d'erreurs total visé et non le
taux ajouté, si bien que --bruit 1 laisse les reads intacts.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import List, Optional, Tuple
import math
import queue
import sys
import threading
import time
import os

import numpy as np
from tqdm import tqdm

import fna_reader
import kmer_tools
import noise
import snp_assembly

DNA_1_PATH = "data/salmonella-enterica.reads.fna"
DNA_2_PATH = "data/salmonella-enterica-variant.reads.fna"

DEFAULT_BATCH_SIZE = 25000        # nombre de reads envoyés par tâche à chaque worker
MAX_SUBSTITUTIONS = 3             # jusqu'à combien de substitutions par k-mère chercher
MAX_CANDIDATES_NEXT_LEVEL = 5000  # au-delà, on n'essaie pas le niveau suivant
CONTEXT_LENGTH = 150              # longueur visée pour la séquence entourant un SNP
MAX_UNMATCHED_DECODED = 50000     # au-delà, on ne recolle pas les candidats sans paire
PREFILTER_CUTOFF = 1              # k-mères vus une seule fois : écartés dès le comptage
LOST_KMERS_BUDGET = 1e-3          # k-mères du génome que le nettoyage s'autorise à perdre
IN_FLIGHT_PER_WORKER = 2          # lots en attente au plus par worker (borne la mémoire)
STEP_LABEL_WIDTH = 44             # largeur des libellés "x/y : ..." pour aligner les barres
NOISE_SEED_STRIDE = 1000003       # écart entre les graines de deux lots successifs


# --- AFFICHAGE ---

def progress_bar(step: str, total: int, unit: str, unit_scale: bool = False) -> tqdm:
    """
    Renvoie une barre de progression au format commun à toutes les étapes, de
    façon que les libellés et les barres restent alignés d'une étape à l'autre.
    """
    return tqdm(
        total=total,
        desc=step.ljust(STEP_LABEL_WIDTH),
        unit=unit,
        unit_scale=unit_scale,
        smoothing=0.05,
        mininterval=0.5,
        bar_format="{desc}{percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} " + unit + " [{elapsed}<{remaining}]",
    )

def print_time_bars(title: str, durations: dict, reference: float, indent: str = "        "):
    """
    Affiche la décomposition d'une durée en barres, chacune longue à
    proportion de la part qu'une sous-étape occupe dans la durée de référence.
    """
    if reference <= 0:
        return
    print(f"{indent}{title}")
    width = 36
    for name, seconds in durations.items():
        filled = int(round(width * min(seconds / reference, 1.0)))
        bar = "#" * filled + "." * (width - filled)
        print(f"{indent}  {name:<18}|{bar}| {seconds:5.0f}s  {100 * seconds / reference:3.0f}%")


class _Tee:
    """
    Écrit sur plusieurs sorties à la fois.

    Une exécution bruitée produit des milliers de lignes de diagnostic, qui
    chassent le début de l'affichage hors de la console. Le fichier, lui,
    garde tout. Les barres de progression passent par un autre canal et ne
    viennent donc pas le salir.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def print_step(step: str, message: str):
    """
    Affiche une étape qui n'a pas de barre de progression, sur une seule ligne
    alignée avec celles qui en ont une.
    """
    print(f"{step.ljust(STEP_LABEL_WIDTH)}{message}")


# --- COMPTAGE ---

def _noise_then_count(batch: List[bytes], k: int, noise_rate: float, seed: int):
    """
    Bruite un lot de reads puis en compte les k-mères repliés, et renvoie ce
    que kmer_tools.count_kmers_batch renvoie.

    Le bruitage a lieu dans le worker, au plus près du comptage : le fil
    principal ne fait que lire le fichier, et rien de ce qui est bruité ne
    transite entre les processus.
    """
    return kmer_tools.count_kmers_batch(noise.add_noise(batch, noise_rate, seed), k)


def get_count_kmer_in_genome(
    k: int, data_path: str, n_workers: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE, noise_rate: float = 0.0, noise_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compte les k-mères d'un fichier de reads et renvoie les k-mères distincts,
    triés, avec le nombre de fois que chacun a été lu. Ajoute au passage la
    proportion noise_rate d'erreurs artificielles, si elle n'est pas nulle.

    Les reads sont découpés en lots que des workers bruitent, encodent,
    replient et comptent chacun de leur côté ; chaque résultat qui revient est
    aussitôt versé dans un arbre de fusion. Rien n'attend la fin de la lecture,
    ce qui évite de garder en mémoire autant de comptages partiels que de lots.

    Chaque lot reçoit sa propre graine, déduite de noise_seed et de son rang :
    deux exécutions de mêmes paramètres bruitent donc exactement pareil.
    """
    kmer_tools.check_k_supported(k)
    n_workers = n_workers or os.cpu_count() or 1
    filename = os.path.basename(data_path)

    noise_note = f", +{100 * noise_rate:.0f}% d'erreurs ajoutées" if noise_rate > 0 else ""
    print(f"\n--> Comptage des k-mères : {filename} "
          f"(k={k}, {kmer_tools.n_limbs_for_k(k)} limbe(s), {n_workers} workers{noise_note})")

    # 1/2 : pré-comptage des reads, pour donner son total à la barre suivante
    with progress_bar("1/2 : indexation du fichier", fna_reader.file_size(data_path), "o", unit_scale=True) as pbar:
        total_reads = fna_reader.count_fasta_records(data_path, on_progress=pbar.update)

    # 2/2 : lecture par lots, comptage et repliage dans les workers, fusion
    tree = kmer_tools.MergeTree(k)
    max_in_flight = n_workers * IN_FLIGHT_PER_WORKER

    # Chronomètres : le fil principal mesure ce qu'il fait lui-même, les
    # workers renvoient le temps de ce qu'ils font de leur côté.
    chrono = {"lecture": 0.0, "attente": 0.0, "fusion": 0.0,
              "encodage": 0.0, "repliage": 0.0, "tri": 0.0}
    t_step = time.perf_counter()

    # Fusion dans un fil séparé, pour qu'elle se déroule pendant la lecture
    # et la distribution des lots.
    to_merge: "queue.Queue" = queue.Queue(maxsize=max_in_flight)

    def merge_loop():
        """
        Prend les comptages dans la file dès qu'ils arrivent et les verse dans
        l'arbre, pendant que le fil principal continue de lire.
        """
        while True:
            item = to_merge.get()
            if item is None:
                return
            local_kmers, local_counts, n_reads = item
            t0 = time.perf_counter()
            tree.add(local_kmers, local_counts)
            chrono["fusion"] += time.perf_counter() - t0
            pbar.update(n_reads)

    with ProcessPoolExecutor(max_workers=n_workers) as executor, \
            progress_bar("2/2 : lecture, comptage, repliage, fusion", total_reads, "reads") as pbar:
        in_flight = {}  # future -> nombre de reads du lot
        merger = threading.Thread(target=merge_loop, daemon=True)
        merger.start()

        def absorb(done_futures):
            for future in done_futures:
                local_kmers, local_counts, worker_chrono = future.result()
                chrono["encodage"] += worker_chrono["encodage"]
                chrono["repliage"] += worker_chrono["repliage"]
                chrono["tri"] += worker_chrono["tri"]
                # File bornée : si la fusion prend du retard, l'attente a lieu ici.
                t0 = time.perf_counter()
                to_merge.put((local_kmers, local_counts, in_flight.pop(future)))
                chrono["attente"] += time.perf_counter() - t0

        batches = fna_reader.stream_fna_batches(data_path, batch_size=batch_size)
        batch_index = 0
        while True:
            t0 = time.perf_counter()
            batch = next(batches, None)
            chrono["lecture"] += time.perf_counter() - t0
            if batch is None:
                break
            if len(in_flight) >= max_in_flight:
                t0 = time.perf_counter()
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                chrono["attente"] += time.perf_counter() - t0
                absorb(done)
            seed = noise_seed * NOISE_SEED_STRIDE + batch_index
            in_flight[executor.submit(
                _noise_then_count, batch, k, noise_rate, seed)] = len(batch)
            batch_index += 1
        while in_flight:
            t0 = time.perf_counter()
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            chrono["attente"] += time.perf_counter() - t0
            absorb(done)

        to_merge.put(None)
        merger.join()

        # Les tables que l'arbre garde en attente sont réunies une fois la
        # lecture finie : c'est la seule fusion qui porte sur la table entière.
        t0 = time.perf_counter()
        acc_kmers, acc_counts = tree.result()
        chrono["fusion"] += time.perf_counter() - t0

    total_step = time.perf_counter() - t_step
    # Les trois premiers postes sont ce que fait le processus principal : leur
    # somme vaut le temps de l'étape. Les deux suivants sont ce que font les
    # workers en parallèle, sur une autre échelle (leur somme peut dépasser).
    # Le fil principal ne fait que lire et attendre : ces deux postes
    # s'additionnent pour faire le temps de l'étape.
    print_time_bars(
        f"Temps de l'étape ({total_step:.0f}s), fil principal :",
        {"lecture": chrono["lecture"], "attente workers": chrono["attente"]},
        total_step,
    )
    # Les autres postes se déroulent en parallèle : ils se superposent aux
    # précédents au lieu de s'y ajouter.
    print_time_bars(
        "En parallèle, dans les autres fils :",
        {"fusion": chrono["fusion"]},
        total_step,
    )
    worker_total = chrono["encodage"] + chrono["repliage"] + chrono["tri"]
    print_time_bars(
        f"En parallèle, dans les {n_workers} workers ({worker_total:.0f}s cumulées, "
        f"soit {worker_total / n_workers:.0f}s chacun) :",
        {"encodage": chrono["encodage"], "repliage": chrono["repliage"],
         "tri": chrono["tri"]},
        worker_total,
    )

    print(f"Terminé : {len(acc_kmers)} k-mères uniques (repliés).")
    return acc_kmers, acc_counts


# --- NETTOYAGE ---

def kmer_spectrum(counts: np.ndarray) -> np.ndarray:
    """
    Compte, pour chaque nombre d'occurrences, combien de k-mères l'atteignent.

    La courbe obtenue présente deux bosses : les k-mères nés d'erreurs de
    séquençage près de 1, ceux du génome autour de la couverture.
    """
    return np.bincount(counts) if counts.size > 0 else np.zeros(1, dtype=np.int64)

def find_cutoff_local_min(spectrum: np.ndarray) -> int:
    """
    Renvoie le premier creux de la courbe, c'est-à-dire le point où elle cesse
    de descendre. Ce creux sépare la bosse des erreurs de celle du génome.
    """
    for x in range(1, len(spectrum) - 1):
        if spectrum[x] <= spectrum[x + 1]:
            return x
    return -1

def estimate_coverage(spectrum: np.ndarray, cutoff: int) -> int:
    """
    Renvoie la couverture typique d'un k-mère du génome : le sommet de la
    bosse située à droite du creux, celle de gauche étant occupée par les
    k-mères nés d'erreurs.
    """
    if len(spectrum) <= cutoff + 1:
        return 0
    return int(np.argmax(spectrum[cutoff + 1:]) + cutoff + 1)


def estimate_genome_size(spectrum: np.ndarray, cutoff: int) -> int:
    """
    Renvoie le nombre de k-mères distincts situés au-dessus du creux, qui
    approche le nombre de positions du génome : les k-mères nés d'erreurs
    forment la bosse de gauche et restent en deçà.
    """
    return int(spectrum[cutoff + 1:].sum()) if len(spectrum) > cutoff + 1 else 0


def poisson_cutoff(coverage: int, genome_size: int) -> int:
    """
    Renvoie le plus grand seuil qui fasse perdre moins de LOST_KMERS_BUDGET
    k-mères du génome, ou -1 s'il n'en existe aucun.

    Le nombre de fois qu'un k-mère du génome est lu suit une loi de Poisson de
    moyenne égale à la couverture. Le nombre de k-mères perdus en écartant tous
    ceux vus au plus T fois vaut donc la taille du génome multipliée par la
    fonction de répartition de cette loi en T, quantité que l'on fait croître
    tant qu'elle tient dans le budget.
    """
    if coverage <= 0 or genome_size <= 0:
        return -1

    cumulative, term, cutoff = 0.0, math.exp(-coverage), -1
    for occurrences in range(coverage + 1):
        cumulative += term
        if genome_size * cumulative > LOST_KMERS_BUDGET:
            break
        cutoff = occurrences
        term *= coverage / (occurrences + 1)
    return cutoff


def find_common_cutoff(spectrum_1: np.ndarray, spectrum_2: np.ndarray) -> Tuple[int, int]:
    """
    Renvoie, à partir des spectres des deux souches, un seuil unique pour les
    deux et la couverture estimée.

    Les deux courbes sont additionnées et le creux est cherché une seule fois
    sur cette somme. Cherché séparément, il tomberait sur une valeur différente
    dans chaque souche, et un k-mère lu juste ce nombre de fois serait gardé
    d'un côté et écarté de l'autre : il ressortirait alors comme une différence
    entre les souches, alors qu'il n'en est pas une.

    Le creux est la borne basse, en deçà de laquelle des k-mères d'erreur
    survivraient ; la loi de Poisson donne la borne haute, au-delà de laquelle
    des k-mères du génome seraient perdus. Le seuil rendu est la borne haute,
    et le creux quand la couverture est trop basse pour qu'elle existe.
    """
    total = np.zeros(max(len(spectrum_1), len(spectrum_2)), dtype=np.int64)
    total[:len(spectrum_1)] += spectrum_1
    total[:len(spectrum_2)] += spectrum_2

    local_min = find_cutoff_local_min(total)
    coverage = estimate_coverage(total, max(local_min, 0))
    genome_size = estimate_genome_size(total, max(local_min, 0))
    return max(local_min, poisson_cutoff(coverage, genome_size)), coverage

def clean_kmer_with_errors(kmers: np.ndarray, counts: np.ndarray, max_errors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Écarte les k-mères dont les occurrences ne dépassent pas max_errors, et
    renvoie ceux qui restent avec leurs occurrences.
    """
    mask = counts > max_errors
    return kmers[mask], counts[mask]


def relieve_and_prefilter(kmers: np.ndarray, counts: np.ndarray):
    """
    Relève le spectre d'un comptage, puis en écarte les k-mères vus une seule
    fois. Renvoie le spectre, les k-mères restants et leurs occurrences.

    Le spectre est relevé avant, car c'est de lui que se déduit le seuil de
    nettoyage et il lui faut la bosse des erreurs. Un k-mère vu une seule fois
    est de toute façon en deçà de ce seuil : le retirer maintenant plutôt
    qu'après libère l'essentiel de la table avant que la souche suivante ne
    soit comptée.
    """
    spectrum = kmer_spectrum(counts)
    kept_kmers, kept_counts = clean_kmer_with_errors(kmers, counts, PREFILTER_CUTOFF)
    return spectrum, kept_kmers, kept_counts


# --- RECHERCHE DE SNPs ---

def _kmer_occurrences(kmers: np.ndarray, counts: np.ndarray, k: int):
    """
    Renvoie une fonction qui dit combien de fois un k-mère, donné en lettres,
    a été lu dans cette souche, et zéro s'il en est absent.

    Passer cette fonction plutôt que la table elle-même permet d'allonger une
    séquence sans rien savoir de l'encodage ni du tri des k-mères.
    """
    def occurrences(kmer: str) -> int:
        index = kmer_tools.find_kmer(kmer_tools.encode_kmer_row(kmer, k), kmers, k)
        return int(counts[index]) if index >= 0 else 0
    return occurrences


def _decode_candidates(candidates: np.ndarray, k: int) -> List[str]:
    """
    Traduit des k-mères en chaînes de lettres, et renvoie la liste vide s'ils
    sont plus de MAX_UNMATCHED_DECODED : au-delà, leur lecture n'apprend plus
    rien et coûte cher.
    """
    if candidates.shape[0] > MAX_UNMATCHED_DECODED:
        return []
    return [kmer_tools.decode_kmer_row(candidates[i], k) for i in range(candidates.shape[0])]


def _swap_direction(snp_pairs: List[dict]) -> List[dict]:
    """
    Échange les champs "dna1" et "dna2" de chaque paire.

    La recherche est menée dans les deux sens ; celles trouvées en partant de
    la seconde souche décrivent donc la même chose à l'envers, et doivent être
    remises dans le sens commun avant d'être regroupées avec les autres.
    """
    return [
        {
            "kmer_dna1": pair["kmer_dna2"], "kmer_dna2": pair["kmer_dna1"],
            "count_dna1": pair["count_dna2"], "count_dna2": pair["count_dna1"],
            "positions": pair["positions"],
            "alleles_dna1": pair["alleles_dna2"], "alleles_dna2": pair["alleles_dna1"],
        }
        for pair in snp_pairs
    ]


def _search_one_direction(kmers_from, counts_from, kmers_to, counts_to, k, max_substitutions):
    """
    Cherche les paires dans un seul sens, par niveaux : d'abord les k-mères
    qui ne diffèrent que d'un nucléotide, puis, sur les seuls candidats restés
    sans paire, ceux qui diffèrent de deux, et ainsi de suite.

    Le travail par candidat croît comme k puissance le niveau, d'où deux
    garde-fous : un niveau maximal, et un nombre maximal de candidats admis au
    niveau suivant. Sans eux la recherche s'éterniserait, car les k-mères nés
    d'erreurs de séquençage n'ont de jumeau à aucun niveau et repartent donc
    intacts d'un niveau au suivant.

    Renvoie les paires rangées par niveau, les candidats restés sans paire, et
    le nombre de candidats de départ.
    """
    only_in_from = ~kmer_tools.is_in_sorted(kmers_from, kmers_to)
    candidates = kmers_from[only_in_from]
    candidate_counts = counts_from[only_in_from]

    pairs_by_level = {}
    remaining, remaining_counts = candidates, candidate_counts
    for level in range(1, max_substitutions + 1):
        if remaining.shape[0] == 0:
            break
        if level > 1 and remaining.shape[0] > MAX_CANDIDATES_NEXT_LEVEL:
            break
        pairs, explained = kmer_tools.find_snp_pairs(
            remaining, remaining_counts, kmers_to, counts_to, k, n_substitutions=level)
        if pairs:
            pairs_by_level[level] = pairs
        remaining, remaining_counts = remaining[~explained], remaining_counts[~explained]

    return pairs_by_level, remaining, candidates.shape[0]


def find_snp_between_clean_kmers(
    kmers_dna1: np.ndarray, counts_dna1: np.ndarray,
    kmers_dna2: np.ndarray, counts_dna2: np.ndarray,
    k: int, max_substitutions: int = MAX_SUBSTITUTIONS,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Identifie les SNPs entre deux tables de k-mères nettoyées, et renvoie les
    paires trouvées, la liste des SNPs et les candidats restés sans paire.

    La recherche est menée dans les deux sens, aucune des deux souches ne
    servant de référence, puis les paires qui décrivent un même SNP sont
    regroupées et leur contexte allongé.
    """
    kmer_tools.check_k_supported(k)
    print(f"\n--> Recherche des SNPs (k={k}, jusqu'à {max_substitutions} substitutions par k-mère)")

    # 1/3 : candidats, puis appariement niveau par niveau
    t0 = time.perf_counter()
    forward = _search_one_direction(kmers_dna1, counts_dna1, kmers_dna2, counts_dna2, k, max_substitutions)
    backward = _search_one_direction(kmers_dna2, counts_dna2, kmers_dna1, counts_dna1, k, max_substitutions)
    pairs_by_level_1, unmatched_1, n_candidates_1 = forward
    pairs_by_level_2, unmatched_2, n_candidates_2 = backward
    search_time = time.perf_counter() - t0

    print_step("1/4 : k-mères absents de l'autre souche",
               f"{n_candidates_1} (dna1) + {n_candidates_2} (dna2)")

    unique_pairs = {}
    for level in sorted(set(pairs_by_level_1) | set(pairs_by_level_2)):
        found = pairs_by_level_1.get(level, []) + _swap_direction(pairs_by_level_2.get(level, []))
        for pair in found:
            unique_pairs.setdefault((pair["kmer_dna1"], pair["kmer_dna2"]), pair)
        print_step(f"2/4 : paires à {level} substitution(s)", f"{len(found)} paires")
    snp_pairs = list(unique_pairs.values())
    print_step("      total",
               f"{len(snp_pairs)} paires, {unmatched_1.shape[0] + unmatched_2.shape[0]} candidats "
               f"sans paire [{search_time:.1f}s]")

    # 3/4 : regroupement des paires décrivant un même SNP
    t0 = time.perf_counter()
    snps = snp_assembly.find_snps_from_pairs(snp_pairs, k)
    print_step("3/4 : regroupement des paires en SNPs",
               f"{len(snps)} SNPs [{time.perf_counter() - t0:.1f}s]")

    # 4/4 : allongement des contextes, une séquence de 2k-1 lettres étant trop
    # courte pour identifier le gène auquel elle appartient
    t0 = time.perf_counter()
    snps = [snp_assembly.extend_snp_context(snp, k, _kmer_occurrences(kmers_dna1, counts_dna1, k),
                                            CONTEXT_LENGTH)
            for snp in snps]
    shortest = min((len(snp["context_dna1"]) for snp in snps), default=0)
    print_step("4/4 : allongement des contextes",
               f"au moins {shortest} pb [{time.perf_counter() - t0:.1f}s]")

    # Diagnostic : les candidats qu'aucun niveau n'a expliqués.
    unmatched = {
        "dna1": _decode_candidates(unmatched_1, k),
        "dna2": _decode_candidates(unmatched_2, k),
    }

    return snp_pairs, snps, unmatched


# --- AFFICHAGE DES RÉSULTATS ---

def print_snp_list(snps: List[dict], k: int):
    """
    Affiche la liste des SNPs, qui est la sortie attendue de l'outil.

    Chaque ligne donne le changement de nucléotide, le nombre de coupes qui
    confirment le SNP sur le nombre possible, la distance à son voisin le plus
    proche, les occurrences moyennes dans chaque souche, puis la séquence qui
    l'entoure avec les deux allèles entre crochets.
    """
    print("\n" + "=" * 45)
    print("                LISTE DES SNPs")
    print("=" * 45)
    if not snps:
        print("Aucun SNP identifié.")
    for i, snp in enumerate(snps, 1):
        index = snp["variant_index"]
        context = (f"{snp['context_dna1'][:index]}"
                   f"[{snp['allele_dna1']}/{snp['allele_dna2']}]"
                   f"{snp['context_dna1'][index + 1:]}")
        neighbour = (f"voisin à {snp['neighbour_distance']:2d}" if snp["n_linked"] > 1
                     else "isolé      ")
        possible = k - snp["n_linked"] + 1  # fenêtres contenant tous les SNPs liés
        print(f"SNP {i:3d} | {snp['allele_dna1']}>{snp['allele_dna2']} "
              f"| {snp['n_cuts']:2d}/{possible} coupes | {neighbour} "
              f"| occ. {snp['mean_count_dna1']}/{snp['mean_count_dna2']}")
        print(f"        contexte ({len(snp['context_dna1'])} pb) : {context}")
    print("=" * 45)

def print_unmatched_regions(unmatched: dict, k: int):
    """
    Recolle les candidats restés sans paire et affiche les séquences obtenues,
    une souche après l'autre.

    Un candidat sans paire signale une différence que la recherche n'explique
    pas par des substitutions : lire les deux souches l'une sous l'autre est
    ce qui permet d'en juger.
    """
    print("\n" + "=" * 45)
    print("      ZONES SANS CORRESPONDANCE (diagnostic)")
    print("=" * 45)
    for name, kmers in (("DNA 1", unmatched["dna1"]), ("DNA 2", unmatched["dna2"])):
        sequences = snp_assembly.assemble_kmers(kmers)
        print(f"{name} : {len(kmers)} k-mères sans paire -> {len(sequences)} zone(s)")
        for i, sequence in enumerate(sequences, 1):
            print(f"  {i}) {len(sequence):4d} lettres : {sequence}")
    print("=" * 45)

# --- PIPELINE COMPLET ---

def prog(k: int = 31, path_dna1: str = DNA_1_PATH, path_dna2: str = DNA_2_PATH,
         max_substitutions: int = MAX_SUBSTITUTIONS, batch_size: int = DEFAULT_BATCH_SIZE,
         error_percent: int = noise.BASE_ERROR_RATE):
    """
    Déroule le pipeline entier sur deux fichiers de reads et affiche les SNPs
    trouvés : compilation, comptage des k-mères, nettoyage, comparaison des
    souches, puis rapport. error_percent donne le taux d'erreurs total visé, en pourcent, et
    des erreurs artificielles sont ajoutées pour l'atteindre.

    C'est la seule fonction qui connaisse l'enchaînement complet ; tout ce
    qu'elle appelle ignore ce qui vient avant et après.
    """
    noise_rate = noise.noise_rate_for(error_percent)
    t_start = time.perf_counter()
    print("--> Début du pipeline")
    if noise_rate > 0:
        print(f"    Robustesse : reads bruités pour atteindre {error_percent}% d'erreurs")

    # 0. Compilation. La faire ici, une fois, plutôt que de la laisser se
    #    déclencher dans les huit workers à la fois au premier fichier compté.
    t0 = time.perf_counter()
    kmer_tools.compile_kernels(k)
    print(f"\n--> Compilation Numba : {time.perf_counter() - t0:.0f}s "
          f"(refaite seulement après une modification du code)")

    # 1. Comptage : une graine par souche, pour que les deux soient bruitées
    #    différemment tout en restant reproductibles. Le spectre de chacune est
    #    relevé aussitôt, puis sa table allégée des k-mères vus une seule fois.
    kmers_dna1, counts_dna1 = get_count_kmer_in_genome(
        k, path_dna1, batch_size=batch_size, noise_rate=noise_rate, noise_seed=1)
    spectrum_dna1, kmers_dna1, counts_dna1 = relieve_and_prefilter(kmers_dna1, counts_dna1)

    kmers_dna2, counts_dna2 = get_count_kmer_in_genome(
        k, path_dna2, batch_size=batch_size, noise_rate=noise_rate, noise_seed=2)
    spectrum_dna2, kmers_dna2, counts_dna2 = relieve_and_prefilter(kmers_dna2, counts_dna2)

    # 3. Filtrage du bruit
    print("\n--> Nettoyage des k-mères")
    cutoff, coverage = find_common_cutoff(spectrum_dna1, spectrum_dna2)
    cutoff = cutoff if cutoff > 0 else 2

    clean_kmers_dna1, clean_counts_dna1 = clean_kmer_with_errors(kmers_dna1, counts_dna1, cutoff)
    clean_kmers_dna2, clean_counts_dna2 = clean_kmer_with_errors(kmers_dna2, counts_dna2, cutoff)

    print(f"Couverture estimée : {coverage}x")
    print(f"Seuil commun aux deux souches : {cutoff}")
    print(f"DNA 1 : {len(clean_kmers_dna1)}/{int(spectrum_dna1.sum())} k-mères conservés")
    print(f"DNA 2 : {len(clean_kmers_dna2)}/{int(spectrum_dna2.sum())} k-mères conservés")

    # 4. Recherche de SNPs
    snp_pairs, snps, unmatched = find_snp_between_clean_kmers(
        clean_kmers_dna1, clean_counts_dna1, clean_kmers_dna2, clean_counts_dna2,
        k=k, max_substitutions=max_substitutions
    )

    # 5. Statistiques et affichage
    print_snp_list(snps, k)                # sortie de l'outil : la liste des SNPs
    print_unmatched_regions(unmatched, k)  # diagnostic : ce qui n'est pas un SNP

    print(f"\n--> Fin du pipeline ({time.perf_counter() - t_start:.1f}s)")


if __name__ == "__main__":
    # Garde indispensable avec multiprocessing : sans elle, chaque worker
    # ré-importerait ce module et relancerait le pipeline.
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Détecte les positions du génome où deux souches d'une même espèce portent\n"
            "un nucléotide différent, à partir de leurs seuls fichiers de reads.\n\n"
            "  python3 snpDetect.py readsA.fasta readsB.fasta"),
        epilog=(
            "Les valeurs par défaut conviennent aux données de l'étude. Les intervalles\n"
            "d'intérêt indiquent jusqu'où il est raisonnable de s'en écarter."))

    parser.add_argument("readsA", nargs="?", default=DNA_1_PATH,
                        help="fichier FASTA de reads de la première souche")
    parser.add_argument("readsB", nargs="?", default=DNA_2_PATH,
                        help="fichier FASTA de reads de la seconde souche")

    parser.add_argument(
        "--k", type=int, default=31,
        help="longueur des k-mères, c'est-à-dire la taille du découpage des reads.\n"
             "Un k plus grand distingue mieux les positions du génome mais coûte\n"
             "plus cher en temps comme en mémoire.\n"
             "[défaut : 31, pris dans l'intervalle du modèle statistique et sous la\n"
             "limite de 32 au-delà de laquelle un k-mère occupe deux mots machine ;\n"
             "intervalle d'intérêt : 13 à 99, les valeurs extrêmes servant à\n"
             "éprouver la méthode]\n ")
    parser.add_argument(
        "--substitutions", type=int, default=MAX_SUBSTITUTIONS,
        help="nombre maximal de substitutions qu'un même k-mère peut porter, qui\n"
             "règle la sensibilité de la recherche. Avec 1, une mutation faite de\n"
             "trois substitutions voisines reste invisible. Le travail croît comme\n"
             "k puissance ce nombre.\n"
             f"[défaut : {MAX_SUBSTITUTIONS}, intervalle d'intérêt : 1 à 5]\n ")
    parser.add_argument(
        "--lot", type=int, default=DEFAULT_BATCH_SIZE,
        help="nombre de reads confiés d'un coup à un cœur. Plus il est grand, moins\n"
             "il y a de fusions à faire et plus le programme est rapide, mais plus il\n"
             "occupe de mémoire ; et réciproquement.\n"
             f"[défaut : {DEFAULT_BATCH_SIZE}, intervalle d'intérêt : 10000 à 100000]\n ")
    parser.add_argument(
        "--sortie", metavar="CHEMIN", default=None,
        help="écrit aussi tout l'affichage dans ce fichier, barres de progression\n"
             "exceptées. La console reste inchangée ; le fichier garde le début de\n"
             "l'exécution, qu'un long diagnostic ferait autrement défiler hors de vue.\n"
             "[défaut : aucun fichier]\n ")
    parser.add_argument(
        "--bruit", type=int, default=noise.BASE_ERROR_RATE,
        help="taux d'erreurs de séquençage total visé, en pourcent, pour éprouver la\n"
             f"méthode. Les données en portent déjà {noise.BASE_ERROR_RATE} (référence Illumina),"
             " donc --bruit 5 en\najoute 4 : le total réel se situe entre 4 et 5 pour cent,"
             " puisqu'on ignore\noù sont les erreurs d'origine.\n"
             f"[défaut : {noise.BASE_ERROR_RATE}, soit les reads intacts ; "
             f"intervalle d'intérêt : 2 à 20]")

    args = parser.parse_args()
    report = open(args.sortie, "w", encoding="utf-8") if args.sortie else None
    console = sys.stdout
    if report is not None:
        sys.stdout = _Tee(console, report)
    try:
        prog(k=args.k, path_dna1=args.readsA, path_dna2=args.readsB,
             max_substitutions=args.substitutions, batch_size=args.lot,
             error_percent=args.bruit)
    except ValueError as erreur:
        parser.error(str(erreur))
    finally:
        sys.stdout = console
        if report is not None:
            report.close()
