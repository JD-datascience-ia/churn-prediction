Version utilisée Python3.12

Objectif :

L’objectif de ce projet est de prédire le churn client (résiliation) à l’aide de modèles de machine learning à partir de données clients.  
Être capable de les identifier en amont constitue un enjeu majeur pour les entreprises, afin de mettre en place des actions de rétention.  

-Dataset

Source : (Kaggle / Telco dataset…)   
Nombre de clients : 7043   

Ce dataset contient des informations sur les clients tel que :
-Informations de facturations  
-Services souscrits  
-Informations contractuelles ( ancienneté, méthode de paiement)

-Méthodologie

Le projet a été réalisé en plusieurs étapes :

1/ Analyse exploratoire (EDA)  

Analyse des distributions  
Identification des incohérences et valeurs manquantes  
Premières observations sur le churn  

2/ Prétraitement des données  
Conversion de TotalCharges en variable numérique  
Encodage des variables catégorielles (one-hot encoding)  
Séparation en jeu d’entraînement et de test avec stratification  

3/ Modèles de base (baseline)

Trois modèles ont été entraînés :

Régression logistique  
Random Forest  
XGBoost  

Les performances initiales sont identiques entre les modèles.

-Features Engineering  

Plusieurs variables ont été créées afin de mieux capturer le comportement des clients et mieux identifier le risque de churn :  

tenure_group : regroupement des clients selon leur ancienneté  
high_charges : indicateur de charges élevées  
engagement_score : interaction entre ancienneté et charges  
service_count : nombre de services souscrits  

-Résultats  

Modèles de base :  

Recall : 0,80  
Precision : 0.49  
F1-score : 0.61  

Après features engineering :  

Logisitic Regression:  

Recall : 0,79  
Precision : 0.49  
F1-score : 0.61  

RandomForest :  

Recall : 0,81  
Precision : 0.50  
F1-score : 0.62  

XGBoost:  

Recall : 0,83  
Precision : 0.48  
F1-score : 0.61  

-Interprétation

Le feature engineering n’a pas permis d’améliorer significativement les performances globales des modèles.  

Cela suggère que :  

les variables initiales contiennent déjà une grande partie de l’information prédictive.  
les performances sont probablement limitées par la richesse des données disponibles.  

Cependant, certaines différences apparaissent :  

XGBoost maximise le recall, ce qui peut être pertinent si l’objectif est de détecter un maximum de clients susceptibles de churn.  
Random Forest offre un meilleur équilibre, avec un F1-score légèrement supérieur. 

-Amélioration possibles

Feature engineering plus avancé.  
Sélection de variables plus fine.  
Gestion du déséquilibre des classes (SMOTE, ajustement du threshold).  
Interprétabilité des modèles (feature importance, SHAP).  
Déploiement via une application Streamlit.  

-Reproductibilité  

git clone https://github.com/JD-datascience-ia/churn-prediction.git cd churn-prediction 
pip install -r requirements.txt

-Auteur

JD-datascience-ia
