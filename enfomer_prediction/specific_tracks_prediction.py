import argparse
import torch
from enformer_pytorch import from_pretrained
import read_sequence
import pandas as pd
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Enformer-based SNP prediction with specific tracks.")
    
    parser.add_argument("--chrom", type=int, required=True, help="Chromosome number to process.")
    parser.add_argument("--number_snps", type=int, required=True, help="Number of SNPs to process.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the result CSV file.")
    parser.add_argument("--random_data", action="store_true", help="Shuffle SNP data (default: False).")
    parser.add_argument("--custom_range_data", action="store_true", help="Use custom SNPs file instead of default.")
    parser.add_argument("--custom_range_data_frame", type=str, default=None, help="Path to custom SNPs file (required if --custom_range_data is set).")

    return parser.parse_args()

def make_prediction(sequence, model, tracks):
    seq = torch.tensor(sequence)
    prediction = model(seq)
    return sum(prediction['human'][:, tracks].mean(dim=1).cpu().detach().numpy())

def main():
    args = parse_arguments()

    print("Loading model...")
    model = from_pretrained('EleutherAI/enformer-official-rough', target_length=128, dropout_rate=0.1)
    print("Model loaded.")

    # Define tracks of interest
    tracks = [689, 694, 698, 699, 708, 709, 720, 721, 725, 726, 734, 738, 
              749, 759, 762, 765, 768, 782, 784, 785, 789, 797, 802, 805, 809, 818, 819, 
              823, 827, 828, 830, 833, 849, 851, 1105, 1106, 1120, 1121, 1124, 1125, 1130, 
              1131, 1141, 1142, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1157, 1158, 
              1177, 1178, 1180, 1181, 1197, 1198, 1203, 1205, 1209, 1216, 1222, 1228, 1230, 
              1232, 1233, 1388, 1395, 1396, 1399, 1401, 1404, 1413, 1415, 1436, 1438, 1439, 
              1442, 1444, 1446, 1449, 1454, 1461, 1464, 1465, 1466, 1479, 1480, 1484, 1486, 
              1506, 1511, 1515, 1518, 1522, 1524, 1531, 1551, 1552, 1554, 1558, 1559, 1564, 
              1568, 1570, 1575, 1583, 1584, 1586, 1587, 1589, 1595, 1600, 1609, 1614, 1616, 
              1618, 1623, 1624, 1626, 1630, 1633, 1643, 1651, 1653, 1658, 1660, 1668, 1670, 
              1684, 1689, 1690, 1694, 1696, 1701, 1702, 1703, 1704, 1705, 1710, 1714, 1720, 
              1725, 1727, 1733, 1759, 1761, 1764, 1765, 1768, 1772, 1774, 1777, 1780, 1781]

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
        difference = make_prediction(seq1, model, tracks) - make_prediction(seq2, model, tracks)
        results.append(difference)

    range_data.loc[:, 'Result'] = results

    # Save results
    range_data.to_csv(args.result_path, sep='\t', index=False)
    print(f"Data saved to {args.result_path}")

if __name__ == "__main__":
    main()