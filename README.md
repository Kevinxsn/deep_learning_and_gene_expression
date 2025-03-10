# Incorporate Deep Learning Model to Better Predict Individual Gene Expression

## Project Overview
This repository provides a pipeline for **Enformer**, a deep learning model for **predicting gene expression from DNA sequences**. The project involves:
- **Preprocessing genomic data** (VCF, BED files)
- **Generating predictions** for new DNA sequences
- **Evaluating model performance** using correlation metrics

## Objectives
- Evaluate the ability of pre-trained Enformer model to predict individual gene expression levels
- Compare prediction results from Enformer and Polygenic Risk Scores on selected genes across samples to assess their relative effectiveness
- Investigate how SNPs affect gene expression within the APOE locus using deep learning.
- Compare eQTL results with deep learning predictions to validate transcriptional impact assessments.
- Explore the integration of Polygenic Risk Scores (PRS) with machine learning for enhanced AD risk prediction

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
- `normalize_predictions.py` - Normalizes the `'true'` and `'mean_prediction'` for the specified summary files.
- `Analysis.ipynb` - Builds a Stacked model using the predictions from multiple individuals and evaluates performance.

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


# **Part 1: Gene Expression Prediction using Pre-Trained Enformer**

## **Overview**
This section describes how to use the pre-trained Enformer model to **predict gene expression** and analyze the impact of genetic variation. Our approach involves:
- **Using the reference genome (hg38) as input** for baseline predictions.
- **Replacing specific genomic regions** with individual-specific genotypes derived from **GTEx_v8_genotype_EUR_HM3** PLINK data.
- **Comparing predictions** Evaluate Enformer predictions with measured gene expression values.

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
| `gene_prediction.py` | Computes the difference between predicted expression values of reference and variant sequences. |
| `normalize_predictions.py` |Normalizes the `'true'` and `'mean_prediction'` for the specified summary files.|
|`Analysis.ipynb` | Builds a Stacked model using the predictions from multiple individuals and evaluates performance. |

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

# **Part 2: Identifying Relationships Between SNPs and Gene Expression**

## **Overview**
This section presents how we utilized a **fine-tuned Enformer model** to investigate the relationship between **genetic variation and gene expression**. Specifically, we aimed to:
- Predict gene expression using both **reference genome sequences** and **individual-specific genotypic sequences**.
- Identify **the impact of SNPs** on transcription by comparing Enformer predictions for different genetic backgrounds.
- Validate our findings using **eQTL analysis**, assessing the consistency between deep learning-based predictions and established statistical models.

## **Data Files**
📂 [Download Data](https://drive.google.com/drive/folders/1AtvTrPzwBOiXBU9UnPYDj1_iP2aka46q?usp=sharing)

| Filename | Description |
|----------|-------------|
| `reference.fa` | Reference genome (hg38) used for baseline predictions. |
| `GTEx_v8_genotype_EUR_HM3.bed` | PLINK genotype data containing SNPs from GTEx. |
| `GTEx_v8_genotype_EUR_HM3.bim` | SNP annotation file with genomic positions. |
| `GTEx_v8_genotype_EUR_HM3.fam` | Sample metadata file. |
| `genotype_sequences.fasta` | DNA sequences reconstructed from individual-specific genotypes. |
| `expression_summary.tsv` | Summary of predicted and actual expression values for selected genes. |

## **Scripts**
| Script | Description |
|--------|-------------|
| `predict.py` | Runs Enformer to predict gene expression for reference and genotype-modified sequences. |
| `compare_predictions.ipynb` | Computes the difference between predicted expression values of reference and variant sequences. |
| `normalize_predictions.py` | Normalizes expression predictions for statistical comparison. |
| `eqtl_analysis.py` | Performs eQTL analysis to validate deep learning predictions. |

## **Prediction Pipeline**
### **1. Prepare Input Sequences**
- Extract **reference genome** sequences from `reference.fa`.
- Convert PLINK genotype data into **modified DNA sequences** using `GTEx_v8_genotype_EUR_HM3.bed`.
- Construct individual-specific sequences where **SNP positions are altered** based on genotype data.
- Save modified sequences as `genotype_sequences.fasta` for Enformer input.

### **2. Run Predictions**
- Use the fine-tuned **Enformer model** to **predict gene expression** for both:
  - The **original reference genome (hg38)**.
  - The **genotype-altered sequences**.
- Store prediction results for all gene expression tracks.

### **3. Compare Predictions**
- Compute **mean expression values** across all tracks.
- Analyze **log fold-change in expression** between reference and genotype-modified sequences.
- Identify SNPs that cause **significant expression changes**.
- Validate key findings using **GTEx eQTL results**, checking if SNPs with high predicted impact align with statistically significant eQTL signals.

### **4. SNP Impact Quantification**
To assess the functional impact of SNPs, we:
- Extracted the **top SNPs affecting gene expression** using Enformer-predicted signals.
- Compared results from:
  - **Full prediction matrix** (all Enformer tracks).
  - **Selected tracks correlated with gene expression** (e.g., chromatin accessibility and histone marks).
- Overlapped these results with **top eQTL SNPs**, identifying **41 shared SNPs** between Enformer’s predictions and statistical eQTL analysis.

## **Evaluation Metrics**
- **Pearson correlation** between predicted and actual GTEx expression values.
- **Log fold-change** in expression for SNP-altered sequences.
- **eQTL validation rate**, assessing how well deep learning predictions align with empirical SNP-expression associations.

## **Results Summary**
- The **Enformer model's SNP impact predictions** largely aligned with traditional eQTL analyses, validating its ability to capture regulatory interactions.
- SNPs identified through **both Enformer and eQTL** were mostly located in **500 kb downstream of the APOE locus**, suggesting regulatory relevance.
- **Full prediction matrix vs. eQTL overlap:** **41 SNPs** matched.
- **Selected prediction tracks vs. eQTL overlap:** **33 SNPs** matched.

These findings indicate that **deep learning models can complement statistical eQTL methods**, helping to pinpoint SNPs with strong regulatory effects.

## **Next Steps**
- Expand the analysis to **disease-associated SNPs**, particularly those influencing gene expression of neurodegenerative risk loci.
- Investigate **cross-tissue regulatory variation**, using GTEx data from multiple tissues.
- Enhance the interpretability of Enformer’s predictions through **functional genomic validation** (e.g., ChIP-seq, CRISPR perturbation assays).

---
