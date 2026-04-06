Version utilisée Python3.12

Objectif :

Ce projet vise à prédire le churn client (résiliation) à partir de données clients afin d’identifier les utilisateurs à risque.

-Dataset

Source : (Kaggle / Telco dataset…)
Nombre de clients : 7043
Variables principales : tenure, contract, monthly charges…

-Méthodologie

Analyse exploratoire des données (EDA)

Préprocessing :

gestion des valeurs manquantes
encodage des variables

Modélisation :

Logistic Regression
Random Forest
XGBoost

Évaluation :

Accuracy
Precision
Recall
F1-score


-Résultats

     Modèle	        Recall	Precision	F1
Logistic Regression	0.80	0.49	   0.61  
Random Forest	    0.80	0.49	   0.61  
XGBoost	            0.80	0.49	   0.61  

 Les performances étant similaires, cela suggère que les limites viennent davantage des données que du modèle.

