import torch
from enformer_pytorch import Enformer
import read_sequence
from enformer_pytorch import from_pretrained
import pandas as pd
from tqdm import tqdm
print('package imported')


model = from_pretrained('EleutherAI/enformer-official-rough', target_length = 128, dropout_rate = 0.1)
print('model loaded')
def make_prediction(sequence):
    seq = torch.tensor(sequence)
    prediction = model(seq)
    return sum(prediction['human'].mean(dim=1).cpu().detach().numpy())




## chromosome
chrom = 2
## number of SNPs
number_snps = 5000
## result file path
result_path = f'/projects/ps-renlab2/sux002/DSC180/local_testing/result/chr{chrom}_numer_{str(number_snps)}_result.csv'
##Shuffle or Not
random_data = True


whole_range_data = pd.read_csv('../../data/snp_sequence_ranges.txt', delimiter='\t')
print('data_loaded')


if random_data == True:
    # Shuffle the entire DataFrame
    whole_range_data = whole_range_data.sample(frac=1, random_state=42).reset_index(drop=True)
    print('data shuffled')

range_data = whole_range_data[whole_range_data['Chrom'] == chrom]

print('data_selected')

if range_data.shape[0] > number_snps:
    range_data = range_data[:number_snps]

print(range_data)


results = []


for index, row in tqdm(range_data.iterrows(), total=range_data.shape[0], desc="Processing Rows"):
    seq1, seq2 = read_sequence.modify_sequence(row['Allele1'], row['Allele2'], row['Start'], row['End'], row['Position'], str(row['Chrom']))
    difference = make_prediction(seq1) - make_prediction(seq2)
    results.append(difference)

range_data.loc[:, 'Result'] = results


range_data.to_csv(result_path, sep='\t', index=False)
print('data saved')