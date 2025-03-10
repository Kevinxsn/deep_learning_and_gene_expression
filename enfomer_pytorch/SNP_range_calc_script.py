import pandas as pd

# Define chromosome lengths based on human genome reference
chromosome_lengths = {
    '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
    '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636,
    '9': 138394717, '10': 133797422, '11': 135086622, '12': 133275309,
    '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345,
    '17': 83257441, '18': 80373285, '19': 58617616, '20': 64444167,
    '21': 46709983, '22': 50818468, 'X': 156040895, 'Y': 57227415,
    'MT': 16569
}

# Read the BIM file (assuming no column names in the file)
bim_file_path = "/projects/ps-renlab2/sux002/DSC180/data/ef_md_test_01/DATA/GTEx_v8_genotype_EUR_HM3.bim"  # Replace with your .bim file path
bim_data = pd.read_csv(bim_file_path, delim_whitespace=True, header=None)

# Assign column names to the BIM data
bim_data.columns = ['Chrom', 'SNP', 'GenDist', 'Position', 'Allele1', 'Allele2']

# Ensure Chromosome column is treated as a string (since it may contain X, Y, MT)
bim_data['Chrom'] = bim_data['Chrom'].astype(str)

# Define sequence length for extracting SNP flanking regions
sequence_length = 190668

# Function to compute the sequence range for each SNP
def get_sequence_range(chrom, position):
    chrom_length = chromosome_lengths.get(chrom, None)
    if chrom_length is None:
        return None, None  # Handle unexpected chromosome values
    start = max(1, position - sequence_length // 2)
    end = min(chrom_length, position + sequence_length // 2)
    return start, end

# Apply the function to compute Start and End positions for each SNP
bim_data[['Start', 'End']] = bim_data.apply(lambda row: get_sequence_range(row['Chrom'], row['Position']), axis=1, result_type="expand")

# Save the processed SNP sequence range data
output_path = "../../data/snp_sequence_ranges.txt"
bim_data[['Chrom', 'SNP', 'Position', 'Start', 'End', 'Allele1', 'Allele2']].to_csv(output_path, sep="\t", index=False)