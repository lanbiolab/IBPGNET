# IBPGNET
IBPGNET is a computational framework based on pathway hierarchical relationships, which is used to predict lung adenocarcinoma recurrence and explore the internal regulatory mechanisms of lung adenocarcinoma. In addition, IBPGNET can efficiently integrate different omics data and provide global interpretability.

# File description

GCN_Prediction.ipynb, Pathways link prediction code.
Main.ipynb, main function.
data, The data set required for the main function to run.
deepexplain, Weight calculation methods.

# Model training
First, unzip the TCGA-LUAD.varscan2_snv.rar and TCGA-LUAD_cnv.rar files in the data file, and then execute the main file in turn.
Note:The pathway features learned by GCN are saved to Pathways_Feature.csv in the data folder. If you need to recalculate the feature of the pathway, run the GCN_Prediction.ipynb code
