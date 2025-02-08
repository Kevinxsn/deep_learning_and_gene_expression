# Fine-Tuning Enformer for Gene Expression Prediction

## Project Overview
This repository provides a pipeline for **fine-tuning Enformer**, a deep learning model for **predicting gene expression from DNA sequences**. The project involves:
- **Preprocessing genomic data** (FASTA, VCF, BED files)
- **Fine-tuning Enformer** on specific gene expression datasets
- **Generating predictions** for new DNA sequences
- **Evaluating model performance** using correlation metrics

## 📂 Repository Structure
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

## 🛠 How to Use
1. **Prepare Data:** Convert genotype data to DNA sequences and extract expression data. 
2. **Fine-Tune Model:** Train Enformer on processed data (`test.py`).
3. **Generate Predictions:** Use the trained model to predict gene expression (`pred.py`).
4. **Evaluate Performance:** Measure Pearson correlation between predicted and true values.

## 🚀 Next Steps
- **Expand dataset** to additional chromosomes.
- **Optimize training parameters** for improved performance.
- **Develop a batch inference pipeline** for large-scale genomic data.







