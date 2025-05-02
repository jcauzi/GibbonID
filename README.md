# Investigating self-supervised speech models’ ability to classify animal vocalizations

This repository contains code for **Investigating self-supervised speech models’ ability to classify animal vocalizations: The case of gibbon’s vocal identity**

The project uses **linear probing** to study bioacoustics transfer learning from the latent representations of pre-trained speech, bird or audio models. 

### 📚 **Abstract**
With the advent of pre-trained self-supervised learning (SSL) models, speech processing research is showing increasing interest towards disentanglement and explainability.
Amongst other methods, probing speech classifiers has emerged as a promising approach to gain new insights into SSL models out-of-domain performances.
We explore knowledge transfer capabilities of pre-trained speech models with vocalizations from the closest living relatives of humans: non-human primates.
We focus on classifying the identity of northern grey gibbons (Hylobates funereus) from their calls with probing and layer-wise analysis of state-of-the-art SSL speech models compared to pre-trained bird species classifiers and audio taggers.
By testing the reliance of said models on background noise and timewise information, as well as performance variations across layers, we propose a new understanding of the mechanisms underlying speech models efficacy as bioacoustic tools.


### 🚀 **Usage**
To run the project, you need the following:
- Python 3.7 or higher
- Required dependencies (listed below)

### 📥 **Dataset**
The dataset used in this study contains vocalization recordings of female Northern Grey Gibbons. To replicate the experiments or for access to the data, please contact the authors directly. The data is not publicly available for distribution but can be provided upon request.

### 📂 **File Structure**
- `data/` — Folder containing the dataset (for replication purposes, accessible by request)
- `models/` — Pre-trained model or checkpoints used for the study
- `notebooks/` — Jupyter notebooks for running the linear probing analysis
- `scripts/` — Python scripts for preprocessing, training, and evaluation
- `requirements.txt` — List of required Python packages and dependencies

### 📬 **Contact**

For any questions, or if you'd like access to the dataset, please contact the authors at [jules.cauzinille@lis-lab.fr].

---

**Note:** Ensure that all references to directories, files, and scripts match the actual contents of your repository. You can expand or modify sections as needed, based on how the code is structured or the specific details of the repository.
