# IBPGNET
Z Xu, H Liao, L Huang, Q Chen, W Lan, S Li. IBPGNET: lung adenocarcinoma recurrence prediction based on neural network interpretability. Briefings in Bioinformatics 25 (3), bbae080

IBPGNET is a computational framework based on pathway hierarchical relationships, which is used to predict lung adenocarcinoma recurrence and explore the internal regulatory mechanisms of lung adenocarcinoma. In addition, IBPGNET can efficiently integrate different omics data and provide global interpretability.

# File description

GCN_Prediction.ipynb, Pathways link prediction code.  
Main.ipynb, Main function.  
data, The data set required for the main function to run.  
deepexplain, Weight calculation methods.  


# Running environment

Main.ipynb:  
python==3.6.2  
pandas == 1.1.5  
numpy==1.19.5  
networkx==2.5.1  
scikit-learn == 0.24.2  
keras==2.2.4  
tensorflow==1.12.0   

GCN_Prediction.ipynb:  
python==3.7.12  
pandas == 1.3.5   
networkx==2.5.1   
numpy==1.21.6  
matplotlib==3.5.1  
torch==1.11.0  
torch-geometric==2.0.4  
scikit-learn==1.0.2 

# Model training
First, unzip the TCGA-LUAD.varscan2_snv.rar and TCGA-LUAD_cnv.rar files in the data file, and then execute the main file in turn.  
Note:The pathway features learned by GCN are saved to Pathways_Feature.csv in the data folder. If you need to recalculate the feature of the pathway, run the GCN_Prediction.ipynb code  



