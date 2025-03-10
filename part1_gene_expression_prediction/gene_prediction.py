import torch
import pandas as pd
import numpy as np
from enformer_pytorch import from_pretrained
from pyfaidx import Fasta

def get_sequence(start, end):
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

def convert_genotype_to_allele(row):
    """Convert genotype (0/0, 0/1, 1/1) into nucleotide sequences for all individuals."""
    df = pd.read_csv('./data_/chr22_expression.bed', sep='\t')
    exp_ids = df.iloc[:,4:].columns
    ids = pd.read_csv('./data_/GTEx_v8_genotype_EUR_HM3_exclude_dups.allchr.reorder.fam', header=None, sep='\s+').iloc[:,1].tolist()
    intersected_ids = np.intersect1d(ids, exp_ids)

    ref, alt = row["REF"], row["ALT"]
    alleles = {}

    for individual in row.index:
        if individual not in intersected_ids:
            continue 
        genotype = row[individual]
        if genotype == "0/0":
            alleles[individual] = ref  
        elif genotype in ["0/1", "1/0"]:
            alleles[individual] = alt 
        elif genotype == "1/1":
            alleles[individual] = alt 
        else:
            alleles[individual] = "N" 

    return pd.Series(alleles)

def apply_variants(seq, allele_df,region_start):
    """Modify reference sequence based on genotype data for all individuals."""
    df = pd.read_csv('./data_/chr22_expression.bed', sep='\t')
    exp_ids = df.iloc[:,4:].columns
    ids = pd.read_csv('./data_/GTEx_v8_genotype_EUR_HM3_exclude_dups.allchr.reorder.fam', header=None, sep='\s+').iloc[:,1].tolist()
    intersected_ids = np.intersect1d(ids, exp_ids)
    seq_list = list(seq)  
    modified_sequences = {}
    for individual in intersected_ids:
        seq_copy = seq_list.copy()
        for i, row in allele_df.iterrows():
            pos = row["POS"] 
            allele = row[individual]
            if region_start <= pos < region_start + len(seq_copy):
                idx = pos - region_start 
                seq_copy[idx] = allele
        modified_sequences[individual] = "".join(seq_copy)
    return modified_sequences

def one_hot_encode(sequence):
    """Convert a DNA sequence into one-hot encoding (A, C, G, T)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_indices = [mapping.get(base, 4) for base in sequence]
    one_hot = torch.nn.functional.one_hot(torch.tensor(seq_indices), num_classes=5)
    return one_hot.float().T[:4]

def preprocess(gene_idx):
    df = pd.read_csv('./data_/chr22_expression.bed', sep='\t')
    exp_ids = df.iloc[:,4:].columns
    ids = pd.read_csv('./data_/GTEx_v8_genotype_EUR_HM3_exclude_dups.allchr.reorder.fam', header=None, sep='\s+').iloc[:,1].tolist()
    intersected_ids = np.intersect1d(ids, exp_ids)

    gene_idx = df[df['gene_id'] == gene_idx].index.tolist()[0]
    gene_idx

    vcf_file = "./data_/genotype_sequences2.vcf"
    vcf_df = pd.read_csv(vcf_file, comment="#", sep="\t", header=None)
    vcf_columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + ids
    vcf_df.columns = vcf_columns

    allele_df = vcf_df.apply(convert_genotype_to_allele, axis=1)
    allele_df.insert(0, "POS", vcf_df["POS"])

    x = pd.DataFrame()
    x['ids'] = intersected_ids
    x["tss"] = pd.Series([df["start"].iloc[gene_idx]]*len(intersected_ids))
    x["start_enformer"] = x["tss"] - 98000
    x["end_enformer"] = x["tss"] + 98000
    x["start_enformer"] = x["start_enformer"].clip(lower=0)

    seq = get_sequence(x['start_enformer'].iloc[0], x['end_enformer'].iloc[0])
    mod_seq = fix_sequence_length(seq)
    x["sequence"] = pd.Series([mod_seq]*len(intersected_ids))
    modified_seqs = apply_variants(x['sequence'].iloc[0], allele_df, x['start_enformer'].iloc[0])
    x['modified_seq'] = modified_seqs.values()
    x["one_hot"] = x["modified_seq"].apply(lambda seq: one_hot_encode(seq))
    return x


def run_enformer(gene_idx):
    x = preprocess(gene_idx)
    device = "cpu"
    model = from_pretrained('EleutherAI/enformer-official-rough').to(device)
    with torch.no_grad():
        predictions = []
        for i in range(len(x["one_hot"])):
            one_hot_seq = x["one_hot"].iloc[i]
            seq_input = one_hot_seq.unsqueeze(0).to(device)
            seq_input = seq_input.permute(0, 2, 1)
            output = model(seq_input) 
            result = output["human"].cpu().numpy()
            result2 = np.mean(result, axis=1)
            predictions.append(result2) 
    return pd.DataFrame({'output': predictions})