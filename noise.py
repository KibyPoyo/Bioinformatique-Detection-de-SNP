"""
Script permettant d'ajouter des erreurs de séquençage artificielles à des
reads, pour mesurer jusqu'à quel taux d'erreur le pipeline retrouve encore les
SNPs.

Travaille sur des reads bruts, tels que fna_reader les livre : ne lit aucun
fichier et n'écrit rien sur disque.

Le bruit imite celui du séquenceur Illumina : chaque nucléotide a la même
probabilité d'être remplacé par l'un des trois autres, et rien n'est jamais
inséré ni supprimé.

Contraintes : les données portent déjà BASE_ERROR_RATE d'erreurs, donc demander
ce niveau-là n'ajoute rien ; à taux égal, une même graine redonne exactement les
mêmes reads ; une lettre autre que A, C, G ou T est laissée telle quelle.
"""
from typing import List

import numpy as np

import nucleotide_2bits as n2b

BASE_ERROR_RATE = 1   # erreurs déjà présentes dans les données, en pourcent
MAX_ERROR_RATE = 100  # au-delà, la demande n'a plus de sens

LETTER_BY_CODE = np.frombuffer(b"ACGT", dtype=np.uint8)

def check_noise_supported(target_percent: int) -> None:
    """
    Vérifie que le taux d'erreurs demandé est atteignable : on ne sait pas
    descendre sous les erreurs déjà présentes, ni dépasser la totalité.
    """
    if not BASE_ERROR_RATE <= target_percent <= MAX_ERROR_RATE:
        raise ValueError(
            f"bruit={target_percent} invalide : le taux d'erreurs total visé va "
            f"de {BASE_ERROR_RATE} à {MAX_ERROR_RATE} pour cent.")


def noise_rate_for(target_percent: int) -> float:
    """
    Renvoie la proportion d'erreurs à ajouter pour qu'un jeu de reads atteigne
    target_percent pour cent d'erreurs au total. Les données en portent déjà
    BASE_ERROR_RATE, donc ce niveau-là renvoie zéro et laisse les reads intacts.
    """
    check_noise_supported(target_percent)
    return (target_percent - BASE_ERROR_RATE) / 100.0


def add_noise(reads: List[bytes], rate: float, seed: int) -> List[bytes]:
    """
    Renvoie les mêmes reads, chaque nucléotide ayant la probabilité rate
    d'avoir été remplacé par l'un des trois autres, tirés à parts égales.

    Les reads sont bruités d'un seul tenant : à ce volume, une boucle Python
    coûterait plus cher que le comptage qu'elle précède.
    """
    if rate <= 0.0 or not reads:
        return reads

    data = np.frombuffer(b"".join(reads), dtype=np.uint8).copy()
    generator = np.random.default_rng(seed)

    positions = np.flatnonzero(generator.random(data.size) < rate)
    codes = n2b.CODE_BY_BYTE[data[positions]]
    keep = codes != n2b.INVALID_CODE  # ni A, ni C, ni G, ni T : on n'y touche pas
    positions, codes = positions[keep], codes[keep]

    # Le rang tiré désigne l'une des trois autres lettres : on le décale d'un
    # cran dès qu'il atteint la lettre d'origine, qui n'est donc jamais rendue.
    ranks = generator.integers(0, 3, positions.size, dtype=np.uint8)
    data[positions] = LETTER_BY_CODE[ranks + (ranks >= codes)]

    noised, ends = data.tobytes(), np.cumsum([len(read) for read in reads])
    return [noised[end - len(read):end] for read, end in zip(reads, ends)]
