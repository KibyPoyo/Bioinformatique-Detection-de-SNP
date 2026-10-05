"""
Script permettant de coder un nucléotide sur 2 bits et un k-mère sur 2k bits,
pour réduire la place qu'ils prennent en mémoire et les rendre comparables
aussi vite que des entiers.

Travaille sur un nucléotide et sur un k-mère vu comme un seul entier ; ignore
la façon dont cet entier est rangé, les tableaux et les fichiers.

Les fonctions les moins évidentes :
    alternative_code  l'un des trois nucléotides différents d'un autre
    shift_of          décalage amenant un nucléotide donné en bout

Le codage : A vaut 00, C vaut 01, G vaut 10, T vaut 11, et un k-mère est
l'entier formé de ses nucléotides mis bout à bout, le premier dans les bits de
poids fort. Le lire de gauche à droite revient à lire son entier du bit le plus
lourd au plus léger.

Contraintes : seules les lettres ACGT ont un code, toute autre reçoit
INVALID_CODE ; les fonctions njit s'appellent depuis du code compilé.
"""
import numpy as np
from numba import njit

NUCLEOTID = ['A', 'C', 'G', 'T']
NUCLEOTIDE_BITS = 2        # bits occupés par une lettre
NUCLEOTIDE_MASK = 0b11     # masque d'une lettre
INVALID_CODE = 255         # code rendu par toute lettre qui n'est pas ACGT

NUCLEOTID_MAP = {
    ord('A'): 0b00, ord('a'): 0b00,
    ord('C'): 0b01, ord('c'): 0b01,
    ord('G'): 0b10, ord('g'): 0b10,
    ord('T'): 0b11, ord('t'): 0b11,
}

# Table octet ASCII -> code, pour traduire une séquence entière d'un coup.
CODE_BY_BYTE = np.full(256, INVALID_CODE, dtype=np.uint8)
for _byte, _code in NUCLEOTID_MAP.items():
    CODE_BY_BYTE[_byte] = _code


def kmer_bits(k: int) -> int:
    """Renvoie le nombre de bits qu'occupe un k-mère, à deux par nucléotide."""
    return NUCLEOTIDE_BITS * k

def get_kmer_mask(k: int) -> int:
    """
    Renvoie un nombre dont seuls les 2k bits de poids faible valent 1.
    L'appliquer à un entier efface tout ce qui dépasse la longueur d'un
    k-mère, ce qui arrive dès qu'on en décale un vers la gauche.
    """
    return (1 << kmer_bits(k)) - 1

@njit(cache=True, inline="always")
def shift_of(k: int, pos: int) -> int:
    """
    Renvoie de combien de bits il faut décaler un k-mère vers la droite pour
    amener en bout le nucléotide de rang pos. Le rang 0 est le premier
    nucléotide, celui des bits de poids fort.
    """
    return NUCLEOTIDE_BITS * (k - 1 - pos)

def decode_nucleotide(nuc_code: int) -> str:
    """Traduit le code d'un nucléotide en sa lettre : A, C, G ou T."""
    return NUCLEOTID[nuc_code & NUCLEOTIDE_MASK]

def get_nucleotide_at(kmer_code: int, pos: int, k: int) -> int:
    """Renvoie le code du nucléotide de rang pos dans un k-mère encodé."""
    return (kmer_code >> shift_of(k, pos)) & NUCLEOTIDE_MASK

def decode_kmer(kmer_code: int, k: int) -> str:
    """Traduit un k-mère encodé en sa chaîne de k lettres."""
    return "".join(decode_nucleotide(get_nucleotide_at(kmer_code, pos, k)) for pos in range(k))

@njit(cache=True, inline="always")
def complement_code(code: int) -> np.int64:
    """
    Renvoie le code du nucléotide partenaire sur l'autre brin : A avec T,
    C avec G. Les quatre codes étant rangés dans cet ordre, le partenaire
    s'obtient en retranchant le code de 3.
    """
    return np.int64(NUCLEOTIDE_MASK) - np.int64(code)

@njit(cache=True, inline="always")
def alternative_code(code: int, rank: int) -> np.int64:
    """
    Renvoie l'un des trois nucléotides différents de celui passé, désigné par
    rank de 0 à 2. Parcourir les trois rangs énumère donc toutes les
    substitutions possibles d'un nucléotide, sans jamais renvoyer l'original.
    """
    alternative = np.int64(rank)
    if alternative >= code:
        alternative += 1
    return alternative
