import pandas as pd

chromosome_lengths = {
    '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
    '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636,
    '9': 138394717, '10': 133797422, '11': 135086622, '12': 133275309,
    '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345,
    '17': 83257441, '18': 80373285, '19': 58617616, '20': 64444167,
    '21': 46709983, '22': 50818468, 'X': 156040895, 'Y': 57227415,
    'MT': 16569
}

# 读取 BIM 文件（假设没有列名）
bim_file_path = "/projects/ps-renlab2/sux002/DSC180/data/ef_md_test_01/DATA/GTEx_v8_genotype_EUR_HM3.bim"  # 请替换为你的 .bim 文件路径
bim_data = pd.read_csv(bim_file_path, delim_whitespace=True, header=None)

# 添加列名
bim_data.columns = ['Chrom', 'SNP', 'GenDist', 'Position', 'Allele1', 'Allele2']

# 确保 Chrom 是字符串（因为 X, Y, MT 也可能出现）
bim_data['Chrom'] = bim_data['Chrom'].astype(str)

# 定义序列长度
sequence_length = 190668

# 计算每个 SNP 的序列范围
def get_sequence_range(chrom, position):
    chrom_length = chromosome_lengths.get(chrom, None)
    if chrom_length is None:
        return None, None  # 处理异常情况
    start = max(1, position - sequence_length // 2)
    end = min(chrom_length, position + sequence_length // 2)
    return start, end

# 应用函数计算 Start 和 End
bim_data[['Start', 'End']] = bim_data.apply(lambda row: get_sequence_range(row['Chrom'], row['Position']), axis=1, result_type="expand")

# 保存结果
output_path = "../../data/snp_sequence_ranges.txt"
bim_data[['Chrom', 'SNP', 'Position', 'Start', 'End','Allele1', 'Allele2']].to_csv(output_path, sep="\t", index=False)