# Fine-Tuning Enformer for Gene Expression Prediction

## Project Overview
This repository provides a pipeline for **fine-tuning Enformer**, a deep learning model for **predicting gene expression from DNA sequences**. The project involves:
- **Preprocessing genomic data** (FASTA, VCF, BED files)
- **Fine-tuning Enformer** on specific gene expression datasets
- **Generating predictions** for new DNA sequences
- **Evaluating model performance** using correlation metrics

## Repository Structure
### Data Files 
[https://drive.google.com/drive/folders/1AtvTrPzwBOiXBU9UnPYDj1_iP2aka46q?usp=sharing]
- `reference.fa` – Reference genome (hg38) used to extract DNA sequences.
- `genotype_sequences.vcf` – Genotyped variant calls for reconstructing individual-specific sequences.
- `chr22_expression.bed` – Gene expression data for **Chromosome 22**, used as target labels.
- `chr22_dnaseq.fasta` – Reconstructed DNA sequence for **Chromosome 22**, incorporating genetic variants.

### Scripts 
- `data_wrangling.ipynb` – Prepares genomic data, extracts sequences, and processes gene expression values.
- `fine_tune.py` – Fine-tunes Enformer using processed DNA sequences and gene expression targets.
- `pred.py` – Generates predictions using the fine-tuned model and evaluates performance.

### Saved Models & Checkpoints
- `fine_tuned_enformer.pth` – The trained Enformer model after fine-tuning.
- `fine_tuned_seq.pt` – The DNA sequence input used for fine-tuning.
- `fine_tuned_targets.pt` – The corresponding gene expression target tensor.

## How to Use
1. **Prepare Data:** Convert genotype data to DNA sequences and extract expression data. 
2. **Fine-Tune Model:** Train Enformer on processed data (`test.py`).
3. **Generate Predictions:** Use the trained model to predict gene expression (`pred.py`).
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
| `reference.fa` | Reference genome (hg38) used as input for baseline predictions. |
| `GTEx_v8_genotype_EUR_HM3.bed` | PLINK genotype data containing SNPs from GTEx. |
| `GTEx_v8_genotype_EUR_HM3.bim` | SNP annotation file with genomic positions. |
| `GTEx_v8_genotype_EUR_HM3.fam` | Sample metadata file. |
| `genotype_sequences.fasta` | DNA sequences reconstructed from individual-specific genotypes. |

## **Scripts**
| Script | Description |
|--------|-------------|
| `predict.py` | Runs Enformer to predict gene expression for reference and genotype-modified sequences. |
| `compare_predictions.ipynb` | Computes the difference between predicted expression values of reference and variant sequences. |

## **Prediction Pipeline**
### **1. Prepare Input Sequences**
- Extract **reference genome** sequences from `reference.fa`.
- Convert PLINK genotype data into **modified DNA sequences** using `GTEx_v8_genotype_EUR_HM3.bed`.
- Save modified sequences as `genotype_sequences.fasta`.

### **2. Run Predictions**
- Use the fine-tuned Enformer model to **predict gene expression** for both:
  - The **original reference genome (hg38)**.
  - The **genotype-altered sequences**.
- Store prediction results for all gene expression tracks.

### **3. Compare Predictions**
- Compute **mean differences** across all tracks.
- Identify genes where SNP modifications lead to **significant expression changes**.
- Perform additional validation using **eQTL data** from GTEx to assess biological relevance.

## **Evaluation Metrics**
- Pearson correlation between predicted and actual GTEx expression values.
- Log fold-change in expression for SNP-altered sequences.
- Statistical tests for SNPs with the largest impact on expression.

## **Next Steps**
- Extend the prediction analysis to **disease-associated SNPs** linked to protein-coding genes.
- Compare **population-specific genotype effects** on gene expression.
- Integrate experimental validation using **external eQTL datasets**.

---






