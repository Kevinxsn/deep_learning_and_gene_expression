import argparse
import torch
from enformer_pytorch import Enformer, from_pretrained
import read_sequence
import pandas as pd
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Enformer-based SNP prediction.")
    
    parser.add_argument("--chrom", type=int, required=True, help="Chromosome number to process.")
    parser.add_argument("--number_snps", type=int, required=True, help="Number of SNPs to process.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the result CSV file.")
    parser.add_argument("--random_data", action="store_true", help="Shuffle SNP data (default: False).")
    parser.add_argument("--custom_range_data", action="store_true", help="Use custom SNPs file instead of default.")
    parser.add_argument("--custom_range_data_frame", type=str, default=None, help="Path to custom SNPs file (required if --custom_range_data is set).")

    return parser.parse_args()

def make_prediction(sequence, model):
    seq = torch.tensor(sequence)
    prediction = model(seq)
    return sum(prediction['human'].mean(dim=1).cpu().detach().numpy())

def main():
    args = parse_arguments()
    
    print("Loading model...")
    model = from_pretrained('EleutherAI/enformer-official-rough', target_length=128, dropout_rate=0.1)
    print("Model loaded.")

    # Load SNP data
    print("Loading SNP data...")
    if args.custom_range_data:
        if args.custom_range_data_frame is None:
            raise ValueError("Error: --custom_range_data_frame must be specified if --custom_range_data is set.")
        whole_range_data = pd.read_csv(args.custom_range_data_frame, delimiter='\t')
    else:
        whole_range_data = pd.read_csv('../../data/snp_sequence_ranges.txt', delimiter='\t')

    print("Data loaded.")

    # Shuffle data if required
    if args.random_data:
        whole_range_data = whole_range_data.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Data shuffled.")

    # Select relevant chromosome
    range_data = whole_range_data[whole_range_data['Chrom'] == args.chrom]

    print("Data selected.")

    # Limit number of SNPs
    if range_data.shape[0] > args.number_snps:
        range_data = range_data[:args.number_snps]

    print(range_data)

    results = []

    # Iterate through SNPs and make predictions
    for index, row in tqdm(range_data.iterrows(), total=range_data.shape[0], desc="Processing Rows"):
        seq1, seq2 = read_sequence.modify_sequence(row['Allele1'], row['Allele2'], row['Start'], row['End'], row['Position'], str(row['Chrom']))
        difference = make_prediction(seq1, model) - make_prediction(seq2, model)
        results.append(difference)

    range_data.loc[:, 'Result'] = results

    # Save results
    range_data.to_csv(args.result_path, sep='\t', index=False)
    print(f"Data saved to {args.result_path}")

if __name__ == "__main__":
    main()