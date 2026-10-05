"""
Script permettant de recoller des k-mères qui se chevauchent, pour passer de
fragments de quelques dizaines de nucléotides à des séquences lisibles : les
SNPs, leur contexte, et les zones qui diffèrent autrement que par substitution.

Travaille sur des chaînes de nucléotides et sur les dictionnaires décrivant
paires et SNPs ; ne connaît ni l'encodage, ni les tableaux, ni les fichiers.

Les fonctions les moins évidentes :
    find_snps_from_pairs  regroupe les coupes d'un même SNP
    assemble_kmers        recolle des k-mères en séquences
    extend_snp_context    allonge les séquences entourant un SNP

Tout repose sur un seul fait : deux k-mères voisins se chevauchent sur tous
leurs nucléotides sauf un, ce qui permet de les relier de proche en proche.

Contraintes : tous les k-mères reçus ont la même longueur ; une séquence se
lisant dans les deux sens, une seule version est conservée ; l'allongement
interroge une fonction fournie par l'appelant ; tout est écrit en Python.
"""
from typing import Callable, Dict, List, Set, Tuple

NUCLEOTIDES = "ACGT"
DEFAULT_CONTEXT_LENGTH = 150  # longueur de contexte visée, en nucléotides

Occurrences = Callable[[str], int]  # k-mère -> nombre de fois qu'il a été lu


def reverse_complement(sequence: str) -> str:
    """Donne le complément inverse ACGT d'une séquence de nucléotides."""
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


class _Partition:
    """
    Suit des éléments répartis en groupes et sait réunir deux groupes en un
    seul.

    Sert à rassembler de proche en proche les coupes d'un même SNP, sans
    savoir à l'avance combien de groupes il y aura ni lesquels finiront
    ensemble.
    """

    def __init__(self, size: int):
        self.representative = list(range(size))

    def find(self, element: int) -> int:
        """Renvoie le représentant du groupe auquel appartient un élément."""
        while self.representative[element] != element:
            self.representative[element] = self.representative[self.representative[element]]
            element = self.representative[element]
        return element

    def union(self, first: int, second: int) -> None:
        """Réunit en un seul les groupes de deux éléments."""
        self.representative[self.find(first)] = self.find(second)

    def groups(self) -> List[List[int]]:
        """Renvoie les groupes formés, chacun comme la liste de ses éléments."""
        grouped: Dict[int, List[int]] = {}
        for element in range(len(self.representative)):
            grouped.setdefault(self.find(element), []).append(element)
        return list(grouped.values())


# ---------------------------------------------------------------------------
# 1. Des paires de k-mères aux SNPs
# ---------------------------------------------------------------------------

def _mirror_pair(pair: dict, k: int) -> dict:
    """Écrit une paire vue depuis l'autre brin."""
    return {
        "kmer_dna1": reverse_complement(pair["kmer_dna1"]),
        "kmer_dna2": reverse_complement(pair["kmer_dna2"]),
        "count_dna1": pair["count_dna1"],
        "count_dna2": pair["count_dna2"],
        "positions": tuple(k - 1 - p for p in reversed(pair["positions"])),
        "alleles_dna1": reverse_complement(pair["alleles_dna1"]),
        "alleles_dna2": reverse_complement(pair["alleles_dna2"]),
    }


def _group_pairs_by_snp(snp_pairs: List[dict]) -> List[List[int]]:
    """
    Rassemble les paires qui sont les coupes d'un même SNP et renvoie, pour
    chaque groupe, les indices de ses paires.

    Deux coupes se suivent quand leurs k-mères partagent k-1 nucléotides et
    que les positions de substitution reculent d'un nucléotide. De proche en
    proche, toutes les coupes d'un même SNP se retrouvent dans le même groupe.
    """
    groups = _Partition(len(snp_pairs))

    by_start: Dict[Tuple[tuple, str, str], List[int]] = {}
    for index, pair in enumerate(snp_pairs):
        by_start.setdefault(
            (pair["positions"], pair["kmer_dna1"][:-1], pair["kmer_dna2"][:-1]), []).append(index)

    for index, pair in enumerate(snp_pairs):
        if min(pair["positions"]) == 0:
            continue
        following = (tuple(p - 1 for p in pair["positions"]),
                     pair["kmer_dna1"][1:], pair["kmer_dna2"][1:])
        for other in by_start.get(following, []):
            groups.union(index, other)

    return groups.groups()


def _glue_cuts(snp_pairs: List[dict], members: List[int], k: int) -> Tuple[List[int], str, str]:
    """
    Recolle les coupes d'un même SNP en deux séquences, une par souche :
    chaque coupe qui prolonge la séquence lui ajoute un nucléotide. Renvoie
    les indices des coupes employées, puis les deux séquences.

    Une coupe qui ne s'enchaîne pas avec les précédentes est écartée, et son
    indice n'apparaît pas dans le résultat.
    """
    ordered = sorted(members, key=lambda index: -max(snp_pairs[index]["positions"]))
    first = snp_pairs[ordered[0]]
    used, context1, context2 = [ordered[0]], first["kmer_dna1"], first["kmer_dna2"]
    seen_offsets = {first["positions"]}

    for index in ordered[1:]:
        pair = snp_pairs[index]
        if pair["positions"] in seen_offsets:
            continue
        if k > 1 and pair["kmer_dna1"][:-1] != context1[-(k - 1):]:
            continue
        context1 += pair["kmer_dna1"][-1]
        context2 += pair["kmer_dna2"][-1]
        seen_offsets.add(pair["positions"])
        used.append(index)

    return used, context1, context2


def _describe_snp_position(snp_pairs: List[dict], members: List[int], k: int) -> dict:
    """
    Rassemble ce que les coupes d'un même SNP disent de la position du génome
    qu'elles décrivent : la séquence de chaque souche, les positions où les
    deux diffèrent, le nombre de coupes et les occurrences moyennes des
    k-mères.

    Le nombre de coupes mesure la confiance : un SNP vu par une seule coupe
    ne tient qu'à un unique k-mère, donc peut-être à une erreur de séquençage.
    """
    used, context1, context2 = _glue_cuts(snp_pairs, members, k)
    counts1 = [snp_pairs[index]["count_dna1"] for index in used]
    counts2 = [snp_pairs[index]["count_dna2"] for index in used]
    return {
        "context_dna1": context1,
        "context_dna2": context2,
        "variant_indices": snp_pairs[used[0]]["positions"],
        "n_cuts": len(used),
        "mean_count_dna1": round(sum(counts1) / len(counts1)),
        "mean_count_dna2": round(sum(counts2) / len(counts2)),
    }


def _orientation_signature(site: dict) -> Tuple[str, str]:
    """
    Attribue une même étiquette aux deux écritures d'une position du génome
    (à l'endroit et en complément inverse), ce qui permet de les reconnaître
    comme une seule.
    """
    forward = (site["context_dna1"], site["context_dna2"])
    mirror = (reverse_complement(site["context_dna1"]),
              reverse_complement(site["context_dna2"]))
    return min(forward, mirror)


def _keep_best_orientation(sites: List[dict]) -> List[dict]:
    """
    Reçoit chaque SNP en double, décrit une fois dans chaque sens de lecture.
    Renvoie, pour chaque paire de doublons, celui qui compte le plus de coupes.
    """
    best: Dict[Tuple[str, str], dict] = {}
    for site in sites:
        signature = _orientation_signature(site)
        if signature not in best or site["n_cuts"] > best[signature]["n_cuts"]:
            best[signature] = site
    return list(best.values())


def _expand_to_snps(site: dict) -> List[dict]:
    """
    Une même position du génome peut porter plusieurs substitutions voisines.
    Cette fonction en fait autant de SNP distincts, chacun gardant les
    séquences et les coupes de la position, plus sa distance au plus proche de
    ses voisins.
    """
    indices = site["variant_indices"]
    snps = []
    for index in indices:
        neighbours = [abs(index - other) for other in indices if other != index]
        snps.append({
            **site,
            "variant_index": index,
            "allele_dna1": site["context_dna1"][index],
            "allele_dna2": site["context_dna2"][index],
            "n_linked": len(indices),
            "neighbour_distance": min(neighbours) if neighbours else 0,
        })
    return snps


def _snp_flanks(snp: dict) -> List[Tuple[str, str, str, str]]:
    """
    Renvoie les deux écritures d'un SNP, à l'endroit et en complément inverse,
    chacune sous la forme (allèle dna1, allèle dna2, séquence à gauche,
    séquence à droite).

    Séparer ainsi l'allèle de ce qui l'entoure permet de comparer deux SNP
    dont les séquences n'ont pas la même longueur.
    """
    context, index = snp["context_dna1"], snp["variant_index"]
    mirror = reverse_complement(context)
    mirrored_index = len(context) - 1 - index
    return [
        (snp["allele_dna1"], snp["allele_dna2"], context[:index], context[index + 1:]),
        (reverse_complement(snp["allele_dna1"]), reverse_complement(snp["allele_dna2"]),
         mirror[:mirrored_index], mirror[mirrored_index + 1:]),
    ]


def _same_position(first: Tuple[str, str, str, str], second: Tuple[str, str, str, str]) -> bool:
    """
    Dit si deux écritures désignent la même position du génome : les allèles
    coïncident et les séquences voisines se prolongent l'une l'autre, même si
    l'une est plus courte.

    C'est le test qui reconnaît un SNP déjà rencontré lorsqu'il revient avec
    un contexte plus long ou plus court.
    """
    allele1_a, allele2_a, left_a, right_a = first
    allele1_b, allele2_b, left_b, right_b = second
    return (allele1_a == allele1_b and allele2_a == allele2_b
            and (left_a.endswith(left_b) or left_b.endswith(left_a))
            and (right_a.startswith(right_b) or right_b.startswith(right_a)))


def _deduplicate_positions(snps: List[dict]) -> List[dict]:
    """
    Prend une liste de SNP triée du plus confirmé au moins confirmé, et n'en
    garde qu'un par position du génome : le premier, donc le mieux étayé.

    Une même position y figure plusieurs fois, une fois trouvée seule et une
    fois trouvée en même temps que ses voisines.
    """
    kept: List[dict] = []
    kept_flanks: List[Tuple[str, str, str, str]] = []
    for snp in snps:
        flanks = _snp_flanks(snp)
        if any(_same_position(flank, other) for other in kept_flanks for flank in flanks):
            continue
        kept.append(snp)
        kept_flanks.append(flanks[0])
    return kept


def find_snps_from_pairs(snp_pairs: List[dict], k: int) -> List[dict]:
    """
    Regroupe les paires de k-mères (candidats identiques à n substitutions
    près) en différentes coupes (morceaux décalés du même SNP). Les SNPs en
    sortie sont triés par ordre de certitude (du plus au moins certain), en
    fonction du nombre de coupes trouvées.
    """
    both_orientations = snp_pairs + [_mirror_pair(pair, k) for pair in snp_pairs]
    sites = [_describe_snp_position(both_orientations, members, k)
             for members in _group_pairs_by_snp(both_orientations)]

    snps = [snp for site in _keep_best_orientation(sites)
            for snp in _expand_to_snps(site)]
    snps.sort(key=lambda snp: (-snp["n_cuts"], snp["context_dna1"], snp["variant_index"]))
    return _deduplicate_positions(snps)


# ---------------------------------------------------------------------------
# 2. Des candidats sans paire aux zones à lire
# ---------------------------------------------------------------------------

def _index_by(kmers: Set[str], start: int, end: int) -> Dict[str, List[str]]:
    """
    Range des k-mères dans un dictionnaire, en prenant pour clé la tranche de
    nucléotides [start:end] de chacun.

    Retrouver ensuite tous les k-mères qui commencent, ou finissent, par une
    suite donnée devient immédiat, au lieu de parcourir toute la collection.
    """
    index: Dict[str, List[str]] = {}
    for kmer in kmers:
        index.setdefault(kmer[start:end], []).append(kmer)
    return index


def _glue_from(start: str, by_start: Dict[str, List[str]], visited: Set[str]) -> str:
    """
    Part d'un k-mère et allonge la séquence d'un nucléotide chaque fois qu'un
    autre k-mère la prolonge, jusqu'à ce qu'aucun k-mère ne la prolonge plus.
    Note dans visited les k-mères employés, qui ne serviront plus à une autre
    séquence.
    """
    sequence, current = start, start
    visited.add(start)
    while True:
        successors = [kmer for kmer in by_start.get(current[1:], []) if kmer not in visited]
        if not successors:
            return sequence
        current = sorted(successors)[0]  # un seul chemin en cas d'embranchement
        visited.add(current)
        sequence += current[-1]


def _longest_per_orientation(sequences: List[str]) -> List[str]:
    """
    De deux séquences qui se lisent l'une sur l'autre brin, ne garde que la
    plus longue, et renvoie ce qui reste de la plus longue à la plus courte.

    Les k-mères ayant été recollés dans les deux sens, chaque séquence a été
    reconstituée deux fois.
    """
    longest: Dict[str, str] = {}
    for sequence in sequences:
        key = min(sequence, reverse_complement(sequence))
        if key not in longest or len(sequence) > len(longest[key]):
            longest[key] = sequence
    return sorted(longest.values(), key=len, reverse=True)


def assemble_kmers(kmers: List[str]) -> List[str]:
    """
    Recolle en séquences des k-mères décalés d'un nucléotide depuis la
    sélection en entrée. Deux k-mères s'enchaînent quand ils partagent k-1
    nucléotides. Renvoie les séquences de la plus longue à la plus courte.
    """
    if not kmers:
        return []

    both = set(kmers) | {reverse_complement(kmer) for kmer in kmers}
    by_start = _index_by(both, 0, -1)
    by_end = _index_by(both, 1, None)

    starts = [kmer for kmer in both if not by_end.get(kmer[:-1])]
    if not starts:  # aucune extrémité : les k-mères forment une boucle
        starts = sorted(both)[:1]

    visited: Set[str] = set()
    sequences = [_glue_from(start, by_start, visited)
                 for start in sorted(starts) if start not in visited]
    return _longest_per_orientation(sequences)


# ---------------------------------------------------------------------------
# 3. Allongement du contexte d'un SNP
# ---------------------------------------------------------------------------

def _single_continuation(candidates: Dict[str, str], occurrences: Occurrences) -> str:
    """
    Reçoit quatre k-mères candidats, un par nucléotide possible, et cherche
    lesquels ont déjà été comptés. Donne le nucléotide gagnant s'il est seul,
    et rien si aucun ne l'est (la séquence s'arrête) ou si plusieurs le sont
    (elle se divise).
    """
    found = [letter for letter, kmer in candidates.items() if occurrences(kmer) > 0]
    return found[0] if len(found) == 1 else ""


def extend_sequence(sequence: str, k: int, occurrences: Occurrences,
                    target_length: int = DEFAULT_CONTEXT_LENGTH) -> Tuple[str, str]:
    """
    Allonge une séquence jusqu'à target_length en ne lui ajoutant que des
    nucléotides qui forment des k-mères déjà comptés, et renvoie les deux
    morceaux ajoutés, à gauche puis à droite.

    Les deux côtés avancent à tour de rôle, pour que la séquence de départ
    reste au centre. L'allongement s'arrête dès qu'aucun des deux ne peut plus
    avancer.
    """
    left, right = "", ""
    while len(left) + len(sequence) + len(right) < target_length:
        grown = False
        for on_right in (True, False):
            if len(left) + len(sequence) + len(right) >= target_length:
                break
            body = left + sequence + right
            if on_right:
                neighbours = body[-(k - 1):]
                candidates = {letter: neighbours + letter for letter in NUCLEOTIDES}
            else:
                neighbours = body[:k - 1]
                candidates = {letter: letter + neighbours for letter in NUCLEOTIDES}

            letter = _single_continuation(candidates, occurrences)
            if letter:
                grown = True
                right, left = (right + letter, left) if on_right else (right, letter + left)
        if not grown:
            break
    return left, right


def extend_snp_context(snp: dict, k: int, occurrences: Occurrences,
                       target_length: int = DEFAULT_CONTEXT_LENGTH) -> dict:
    """
    Allonge les deux séquences qui entourent un SNP jusqu'à target_length, et
    décale d'autant les positions qu'il porte.

    Les mêmes nucléotides sont ajoutés aux deux souches, puisque leurs
    séquences ne diffèrent qu'aux positions du SNP. Un contexte trop court ne
    permet pas de retrouver le gène touché : c'est ce qui justifie cet
    allongement.
    """
    left, right = extend_sequence(snp["context_dna1"], k, occurrences, target_length)
    if not left and not right:
        return snp

    return {
        **snp,
        "context_dna1": left + snp["context_dna1"] + right,
        "context_dna2": left + snp["context_dna2"] + right,
        "variant_index": snp["variant_index"] + len(left),
        "variant_indices": tuple(i + len(left) for i in snp["variant_indices"]),
    }
