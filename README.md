# Incorporate Deep Learning Model to Better Predict Individual Gene Expression

## Project Overview
This repository provides a pipeline for **Enformer**, a deep learning model for **predicting gene expression from DNA sequences**. The project involves:
- **Preprocessing genomic data** (VCF, BED files)
- **Generating predictions** for new DNA sequences
- **Evaluating model performance** using correlation metrics

## Repository Structure
### Data Files 
- [Genotype and Expression data](https://drive.google.com/drive/folders/1AtvTrPzwBOiXBU9UnPYDj1_iP2aka46q?usp=sharing)
- `38.fa` – Reference genome [(hg38)](https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz) used to extract DNA sequences.
- `genotype_sequences22.vcf` – Genotyped variant calls for reconstructing individual-specific sequences on **Chromosome 22.**
- `chr22_expression.bed` – Gene expression data for **Chromosome 22**, used as target labels.

### Scripts 
- `gtex_prs.ipynb` – Runs PRS on genes on chromosome 22 for comparison with Enformer.
- `individual_prediction.py` – Recontructs sequence for with a specified individual's SNPs and runs the Enformer on the sequence for all genes on chromosome 22.
- `gene_prediction.py` – Recontructs sequence for all individuals' SNPs for a singular gene and runs the Enformer on the sequence for all individuals for the gene.

## How to Use
1. **Installation:** First, install the required dependencies: `pip install torch enformer-pytorch pyfaidx cyvcf2 pandas numpy scipy`
2. **Prepare Data:** Convert genotype data to DNA sequences and extract expression data.
   Extract data for specified chromosome as a Plink file. Throughout this project, we use chromosome 22. 
   ```
   ./plink2 --bfile ./data/GTEx_v8_genotype_EUR_HM3_exclude_dups.allchr.reorder --chr 22 --make-bed --out gtex_chr22
   ```
   Convert Plink file into VCF
   ```
   ./plink2 --bfile ./data/gtex_chr22 --out ./data/genotype_sequences22 --recode vcf
   ```
   Extract chromosome 22 from expression data
   ```
   awk 'NR==1 || $1 == "chr1"' ./data/Whole_Blood.v8.normalized_expression.bed > ./data/chr22_expression.bed
   ```
3. **Generate Predictions:** Use the trained model to predict gene expression (`individual_prediction.py` and `gene_prediction.py`).
4. **Evaluate Performance:** Measure Pearson correlation between predicted and true values.

## Next Steps
- **Expand dataset** to additional chromosomes.
- **Optimize training parameters** for improved performance.
- **Develop a batch inference pipeline** for large-scale genomic data.


# **Gene Expression Prediction using Fine-Tuned Enformer**

## **Overview**
This section describes how to use the fine-tuned Enformer model to **predict gene expression** and analyze the impact of genetic variation. Our approach involves:
- **Using the reference genome (hg38) as input** for baseline predictions.
- **Replacing specific genomic regions** with individual-specific genotypes derived from **GTEx_v8_genotype_EUR_HM3** PLINK data.
- **Comparing predictions** between reference and genotype-altered sequences to assess the effects of SNPs.

## **Data Files**
📂 [Download Data](https://drive.google.com/drive/folders/1AtvTrPzwBOiXBU9UnPYDj1_iP2aka46q?usp=sharing)

| Filename | Description |
|----------|-------------|
| `38.fa` | Reference genome (hg38) used as input for baseline predictions. |
| `GTEx_v8_genotype_EUR_HM3.bed` | PLINK genotype data containing SNPs from GTEx. |
| `GTEx_v8_genotype_EUR_HM3.bim` | SNP annotation file with genomic positions. |
| `GTEx_v8_genotype_EUR_HM3.fam` | Sample metadata file. |


## **Scripts**
| Script | Description |
|--------|-------------|
| `individual_prediction.py` | Runs Enformer to predict gene expression for reference and genotype-modified sequences. |
| `gene_prediction.ipynb` | Computes the difference between predicted expression values of reference and variant sequences. |

## **Prediction Pipeline**
### **1. Prepare Input Sequences**
- Extract chromosomal genotype and expression information from `BED` files. 
- Extract **reference genome** sequences from `38.fa`.
- Apply variants to reference genome using genotype information.
- One hot encode the modified sequences and convert to PyTorch tensor.

### **2. Run Predictions**
- Use the Enformer model to **predict gene expression** for both:
  - The one-hot encoded **genotype-altered sequences**.
- Store prediction results for all gene expression tracks.

### **3. Compare Predictions**
- Compute **mean expression** across all tracks.
- Perform normalization and transform the mean predictions
- Fit a Stacked Model to fine tune the pipeline 

## **Evaluation Metrics**
- Pearson correlation between predicted and actual GTEx expression values.

## **Next Steps**
- Extend the prediction analysis to **disease-associated SNPs** linked to protein-coding genes.
- Compare **population-specific genotype effects** on gene expression.
- Integrate experimental validation using **external eQTL datasets**.

---
