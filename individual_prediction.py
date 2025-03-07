import torch
import pandas as pd
import numpy as np
from enformer_pytorch import from_pretrained
from pyfaidx import Fasta


def convert_genotype_to_allele(row, individual="GTEX-111YS"):
    """Convert genotype (0/0, 0/1, 1/1) into a nucleotide sequence."""
    genotype = row[individual]
    ref, alt = row["REF"], row["ALT"]
    if genotype == "0/0":
        return ref  
    elif genotype in ["0/1", "1/0"]:
        return alt  
    elif genotype == "1/1":
        return alt 
    else:
        return "N"

def get_sequence(chrom, start, end):
    """Fetch DNA sequence from the FASTA file, ensuring valid chromosome names."""
    fasta = Fasta("./data_/38.fa")
    try:
        return str(fasta['22'][start:end])
    except KeyError:
        return "N" * (end - start)

def fix_sequence_length(seq, target_length=196608):
    """Ensure DNA sequence is exactly 196608 bp by padding or trimming."""
    seq = seq.upper()
    if len(seq) < target_length:
        seq += "N" * (target_length - len(seq))
    return seq[:target_length]

def apply_variants(seq, vcf_df, chrom, region_start):
    """Modify reference sequence based on genotype data."""
    seq = list(seq)
    for _, row in vcf_df.iterrows():
        pos = row["POS"]
        allele = row["Allele"]
        if region_start <= pos < region_start + len(seq):
            idx = pos - region_start
            seq[idx] = allele
    return "".join(seq)

def one_hot_encode(sequence):
    """Convert a DNA sequence into one-hot encoding (A, C, G, T)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_indices = [mapping.get(base, 4) for base in sequence]
    one_hot = torch.nn.functional.one_hot(torch.tensor(seq_indices), num_classes=5)
    return one_hot.float().T[:4]

def preprocess(individual_idx):
    ids = pd.read_csv('./data_/GTEx_v8_genotype_EUR_HM3_exclude_dups.allchr.reorder.fam', header=None, sep='\s+').iloc[:,1].tolist()
    vcf_file = "./data_/genotype_sequences2.vcf"
    vcf_df = pd.read_csv(vcf_file, comment="#", sep="\t", header=None)
    vcf_columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + ids # VCF columns
    vcf_df.columns = vcf_columns

    vcf_df["Allele"] = vcf_df.apply(lambda row: convert_genotype_to_allele(row, individual= individual_idx), axis=1)
    q = vcf_df[['POS', 'Allele']]

    df = pd.read_csv('./data_/chr22_expression.bed', sep='\t') # 5ew32wExpression 
    df["tss"] = df["start"]
    df["start_enformer"] = df["tss"] - 98000
    df["end_enformer"] = df["tss"] + 98000
    df["start_enformer"] = df["start_enformer"].clip(lower=0)

    df["sequence"] = df.apply(lambda row: get_sequence(row["#chr"], row["start_enformer"], row["end_enformer"]), axis=1)
    df["sequence"] = df["sequence"].apply(lambda seq: fix_sequence_length(seq))

    df['modified_seq']= df.apply(lambda x: apply_variants(x['sequence'], q, 22, x['start_enformer']), axis=1)

    df["one_hot"] = df["modified_seq"].apply(lambda seq: one_hot_encode(seq))
    return df



def run_enformer(individual_idx):
    df = preprocess(individual_idx)

    device = "cpu"
    model = from_pretrained('EleutherAI/enformer-official-rough').to(device)
    with torch.no_grad():
        predictions = []
        for i in range(len(df["one_hot"])):
            if i%2 == 0:
                print(i)
            one_hot_seq = df["one_hot"].iloc[i]
            seq_input = one_hot_seq.unsqueeze(0).to(device)
            seq_input = seq_input.permute(0, 2, 1)
            output = model(seq_input)
            result = output["human"].cpu().numpy()
            result1 = np.mean(result, axis=1)
            predictions.append(result1)
    return pd.DataFrame({'output': predictions})