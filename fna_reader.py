"""
Script permettant de parcourir un fichier FASTA read par read, sans jamais en
charger plus d'un bloc à la fois, quelle que soit la taille du fichier.

Travaille sur les octets bruts : n'encode rien, n'affiche rien, et signale son
avancement à une fonction que l'appelant fournit. Un fichier FASTA alterne des
en-têtes ouverts par '>' et des lignes de séquence ; seules les secondes
servent au pipeline.

Les fonctions les moins évidentes :
    stream_fna_bytes    séquences des reads, une par une
    stream_fna_batches  les mêmes, groupées en lots pour les workers

Contraintes : un read est coupé par la frontière entre deux blocs, et le
recollage est fait ici ; chaque fonction relit le fichier depuis le début.
"""
import itertools
import os
from typing import Callable, Generator, List, Optional

READ_CHUNK_SIZE = 16 * 1024 * 1024  # taille d'un bloc de lecture


def file_size(filepath: str) -> int:
    """Renvoie la taille d'un fichier en octets."""
    return os.path.getsize(filepath)


def count_fasta_records(filepath: str, on_progress: Optional[Callable[[int], None]] = None) -> int:
    """
    Compte les reads d'un fichier FASTA, repérés par le '>' qui ouvre leur
    en-tête, et signale à on_progress le nombre d'octets de chaque bloc lu.

    Connaître ce total avant la lecture proprement dite est ce qui permet d'en
    afficher l'avancement.
    """
    n_records = 0
    previous_last_byte = b"\n"  # le début du fichier se comporte comme un début de ligne
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            n_records += chunk.count(b"\n>")
            if previous_last_byte == b"\n" and chunk[:1] == b">":
                n_records += 1
            previous_last_byte = chunk[-1:]
            if on_progress is not None:
                on_progress(len(chunk))
    return n_records


def stream_fna_bytes(filepath: str) -> Generator[bytes, None, None]:
    """
    Parcourt un fichier FASTA et renvoie les séquences des reads une par une,
    en-tête et retours à la ligne retirés.

    Aucun autre nettoyage n'est fait : une lettre inattendue reste en place, et
    c'est l'encodage qui la traitera comme un nucléotide invalide.
    """
    leftover = b""
    windows_line_ends = False
    with open(filepath, "rb") as f:
        first_chunk = True
        while True:
            chunk = f.read(READ_CHUNK_SIZE)
            if chunk:
                if first_chunk:
                    windows_line_ends = b"\r\n" in chunk[:4096]
                    first_chunk = False
                # Le dernier morceau du bloc est incomplet : il attend la
                # suite du fichier.
                records = (leftover + chunk).split(b"\n>")
                leftover = records.pop()
            else:
                records, leftover = [leftover], b""

            for record in records:
                # La séquence commence après l'en-tête, donc au premier
                # retour à la ligne.
                header_end = record.find(b"\n")
                if header_end == -1:
                    continue  # en-tête sans séquence
                sequence = record[header_end + 1:].replace(b"\n", b"")
                if windows_line_ends:
                    sequence = sequence.replace(b"\r", b"")
                if sequence:
                    yield sequence

            if not chunk:
                return


def stream_fna_batches(filepath: str, batch_size: int = 20000) -> Generator[List[bytes], None, None]:
    """
    Renvoie les séquences des reads par lots de batch_size, le dernier lot
    pouvant être plus petit.

    Un lot est la quantité de travail confiée d'un coup à un worker : le
    découper ici évite à l'appelant de compter les reads lui-même.
    """
    it = stream_fna_bytes(filepath)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            return
        yield batch
