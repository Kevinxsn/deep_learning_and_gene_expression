from Bio import SeqIO


def char_to_num(sequence):
    mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3, 'N': 4}
    return [mapping[nuc] for nuc in sequence.upper() if nuc in mapping]

def extract_sequence(sequence_id, region_start, region_end, fasta_path = '/projects/ps-renlab2/sux002/DSC180/data/h38.fa'):
    """
    Extracts a sub-region of a DNA sequence from a FASTA file.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    sequence_id : str
        The ID of the sequence to extract (e.g., chromosome name or contig ID).
    region_start : int
        The start position of the sub-region to extract.
    region_end : int
        The end position of the sub-region to extract.

    Returns
    -------
    str
        The extracted DNA subsequence.

    Raises
    ------
    ValueError
        If the specified sequence ID is not found in the FASTA file.
    """
    with open(fasta_path, 'r') as handle:
        fasta_sequences = SeqIO.parse(handle, 'fasta')
        for record in fasta_sequences:
            if record.id == sequence_id:
                # Convert to string and slice the specified region
                return char_to_num(str(record.seq[region_start:region_end]))
        else:
            # If no break occurred, then the ID wasn't found
            raise ValueError(f"Sequence ID '{sequence_id}' not found in the FASTA file.")

# Example usage:
# dna_sequence = extract_sequence(
#     fasta_path="/projects/ps-renlab2/sux002/DSC180/data/h38.fa",
#     sequence_id="1",
#     region_start=10000,
#     region_end=206608
# )
# print(dna_sequence)



def modify_sequence(a1, a2, start, end, position, id):
    sequence = extract_sequence(id, start, end)
    mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3, 'N': 4}
    al_position = position - start
    sequence1 = sequence.copy()
    sequence2 = sequence.copy()
    sequence1[al_position] = mapping[a1]
    sequence2[al_position] = mapping[a2]
    return sequence1, sequence2
    
    
    