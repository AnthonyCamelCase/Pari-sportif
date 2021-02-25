# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:32:22 2021

@author: Anthony
"""

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,make_scorer
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix,confusion_matrix
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option('max_columns', 40)

st.title("Démo Streamlit : pari Sportif")

st.header("Objectif : Prédire le vainqueur du match")

st.sidebar.header("Prédire le vainqueur du match")

# **************chargement donnée********************
st.sidebar.subheader('1-Chargement des données')
st.sidebar.markdown('Variables du jeu de données')


st.subheader('1-Chargement des données')
df = pd.read_csv("data_clean_all.csv",index_col=0)
df= df.drop(["Favori","Challenger","Comment","Location","Date","Series","ATP","Best of","Winner","Wsets","Lsets"],axis=1)
df = df.reset_index().drop("index",axis=1)
st.sidebar.write(df.columns)

if st.checkbox('Afficher les données'):
   st.write(df.head(20))

# **************Séparation feature et target********************
st.sidebar.subheader('2-Séparation feature et target')
st.subheader('2-Séparation feature et target')

with st.echo():
    target = df.Vainqueur
    data =  df.drop(["Vainqueur","PS_F","PS_C","B365_F","B365_C"],axis=1)

# **************Preprocessing********************
st.sidebar.subheader('3-PreProcessing')
 
st.subheader('3-preProcressing')
st.markdown('Variables catégorielles')
with st.echo():
    cat = data.loc[:,data.dtypes==np.object]
    cat = pd.get_dummies(cat)

st.markdown('Variables numériques')
with st.echo():
    num= data.select_dtypes(include='number')
    scaler = StandardScaler()
    num_scaled = pd.DataFrame(scaler.fit_transform(num), columns = num.columns)
    
st.markdown('Features Finales')
with st.echo():
    data = num_scaled.join(cat)

st.write(data.head(10))

# **************train et test********************
st.sidebar.subheader('4-Définition set entrainement et set test')
trainsize = st.sidebar.slider(label = "Choix de la taille de l‘échantillon d'entrainement",
    min_value = 0.2, max_value = 1.0,step = 0.05)

st.subheader('4-Définition set entrainement et set test')

with st.echo():
    X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=trainsize,shuffle=False)
 
# **************algo entrainement et resultat********************

#fonction qui fournit pour un modele donné, après double validation croisée, l'accuracy du modèle, son écart type, le résultat obtenu avec les meilleurs paramètres.
def Grid(clf, param,n,n_jobs):
    outer_cv = StratifiedKFold(n_splits = n, shuffle=True)
    grid_clf = GridSearchCV(estimator=clf,param_grid = param,n_jobs=n_jobs)
    score = cross_val_score(grid_clf,X_train,y_train,cv = outer_cv)
    st.write("accuracy du modèle : ", score.mean())
    st.write("ecart type du modèle : ", score.std())
    grid_clf.fit(X_train,y_train)
    st.write("meilleur paramètre : ", grid_clf.best_params_)
    st.write("score train : ",round(100*grid_clf.score(X_train,y_train),2))
    st.write("score test : ", round(100*grid_clf.score(X_test,y_test),2))
    
    #récupération des probabilités de victoires
    y_pr =grid_clf.predict(data)
    predict = pd.DataFrame(y_pr,index=data.index ,columns=["Prediction"])
    y_pred = grid_clf.predict_proba(data)
    proba = pd.DataFrame(y_pred,index=data.index ,columns=["Proba_Challenger","Pro ba_Favori"])
    
    plot_confusion_matrix(grid_clf,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Challenger","Favori"])
    st.pyplot()

    
#sidebar pour choisir les folds et l'algo
st.sidebar.subheader('5-Etude des différents modèles')
n = st.sidebar.slider(label = 'Selectionnez le nombre de pli pour la validation croisée avec stratifiedKfold',
                      min_value = 2, max_value=10, step=1) 
n_jobs = st.sidebar.slider(label = 'nombre de travaux exécutés en parallèle',
                      min_value = -1, max_value=10, step=2)                    
clf = st.sidebar.selectbox('Selectionnez un algorithme',
                           ["aucun","Logistic Regression", "Random Forest", "Decision Tree", "KNN","XGB"],
                        )

#définition des différents algo et de leur grille de parametre
#Régression logistique
param_lr = {'solver':['liblinear','lbfgs'],'C':np.logspace(-4,2,9)}
clf_lr = LogisticRegression(max_iter= 2000)

#Random Forest
param_rf = [{'n_estimators':[10,50,100,250,500,1000],
             'min_samples_leaf':[1,2,5],
             'max_features':['sqrt','log2']}]
clf_rf = RandomForestClassifier(n_jobs = -1)

#decision tree
param_dtc = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
clf_dtc = DecisionTreeClassifier()

#KNN
param_knn = {'n_neighbors': [3,5,11,19],'weights':['uniform','distance'],'metric':['euclidean','manhattan','minkowski']}
clf_knn = neighbors.KNeighborsClassifier()

#XGBboost
params = {
        'min_child_weight': list(np.arange(1,10,1)),
        'gamma': list(np.arange(0.5,5,0.5)),
        'subsample': list(np.arange(0.5,1,0.1)),
        'colsample_bytree': list(np.arange(0.5,1,0.1)),
        'max_depth': list(np.arange(3,5,1))
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

#entrainement de l'algo et résultats :
st.subheader('5-Résultats de l\'entrainement du modèle choisi')
if (clf == "Logistic Regression"):
    Grid(clf_lr,param_lr,n,n_jobs)
if (clf == "Random Forest"):
    Grid(clf_rf,param_rf,n,n_jobs)
if (clf == "Decision Tree"):
    Grid(clf_dtc,param_dtc,n,n_jobs)
if (clf == "KNN"):
    Grid(clf_knn,param_knn,n,n_jobs) 
if (clf == "XGB"):
    skf = StratifiedKFold(n_splits=n, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs= n, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )
    random_search.fit(X_train, y_train)
    st.write("meilleur paramètre : ", random_search.best_params_)
    st.write("score train : ",round(100*random_search.score(X_train,y_train),2))
    st.write("score test : ", round(100*random_search.score(X_test,y_test),2))
    plot_confusion_matrix(random_search,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Challenger","Favori"])
    st.pyplot()
    feat= pd.Series(random_search.best_estimator_.feature_importances_,index=X_train.columns)
    feat.nlargest(20).plot(kind='barh')
    st.pyplot()

#*********** 2eme ALGO : le ROI ***********

st.title('Le Retour sur Investissement : ROI')
st.sidebar.subheader('Afficher les données avec les indices de confiances')
st.subheader('Données avec les indices de confiances et les gains théoriques')
algo = st.sidebar.selectbox('Selectionnez un modèle et ses probabilités de victoires',
                           ["Logistic Regression", "Random Forest", "Decision Tree", "KNN","XGB", "XGB sans proba_ELO"],)   


#Chargement du fichier csv contenant les probabilités de victoires
if (algo == "Logistic Regression"):
    pari = pd.read_csv("pariLogisticRegression(max_iter=2000).csv",index_col=0)
if (algo == "Random Forest"):
    pari = pd.read_csv("pariRandomForestClassifier(n_jobs=-1).csv",index_col=0)
if (algo == "Decision Tree"):
    pari = pd.read_csv("pariDecisionTreeClassifier().csv",index_col=0)
if (algo == "KNN"):
    pari = pd.read_csv("pariKNeighborsClassifier().csv",index_col=0)
if (algo == "XGB"):
    pari = pd.read_csv("pariXGBoost.csv",index_col=0)
 if (algo == "XGB sans proba_ELO"):
    pari = pd.read_csv("pariXGBoostsansProbaElo.csv",index_col=0)   
    

#ajout de l'indice de confiance sur la cote    s
pari["proba_PS_Challenger"]=1/pari.PS_C
pari["proba_PS_Favori"]=1/pari.PS_F
pari["gain_theoriq_F"] = pari.Proba_Favori*pari.PS_F
pari["gain_theoriq_C"] =  pari.Proba_Challenger*pari.PS_C

#ajout d'une variable avec le meilleur gain théorique 
pari["gain_confiance"]="wait"   
for i in range(len(pari)):
    if pari["Prediction"][i]== "favori":
        pari["gain_confiance"][i]= pari["gain_theoriq_F"][i]
    else:
        pari["gain_confiance"][i]= pari["gain_theoriq_C"][i]
            
#détermination de victoire du pari
pari["pari gagné"]=2
for i in range(len(pari)):
    if pari.Vainqueur[i] == pari.Prediction[i]:
        pari["pari gagné"][i]=1
    else:
        pari["pari gagné"][i]=0
    
#choix de la cote en fonction de la prediction
pari["cote"]=1.1
for i in range(len(pari)): 
    if pari.Prediction[i]=="favori":
        pari.cote[i] = pari.PS_F[i]
    else:
        pari.cote[i] = pari.PS_C[i]
    
#rangement en fonction du gain de confiance        
pari.sort_values(by = 'gain_confiance', ascending = False, inplace = True)
    
#création d'un dataframe test pour pouvoir tester plusieurs parametres essentiellement les cotes ici par exemple
test = pari
#test= test[test.Prediction == test.choix]
#on les range par indice décroissant
test.sort_values(by = 'gain_confiance', ascending = False, inplace = True)
    
test = test.reset_index().drop("index",axis=1)
    


if st.checkbox('Afficher les données avec les indices de confiance calculées'):
   st.write(test.head(50))

st.sidebar.subheader('Résultats obtenus en pariant 1 euros sur n nombre de match')
st.subheader('Résultats obtenus en pariant 1 euros sur n nombre de match')

#fonction calculant le gain pour n match pariés
def gain(n):
    gain = 0.1
    argent = 0.1
    gain_roi = []
    mise_totale = 0
    for i in range(n):
        mise = 1
        if test["pari gagné"][i] == 1:
                argent += mise * test.loc[i,"cote"] -1
        else:
                argent -= mise
    return argent

#fonction calculant le ROI pour n match (c'est la meme fonction qu'au dessus mais je n'arrive pas à récupérer séparemment les valeurs du tuple ensuite)
def Rent(n):
    gain = 0.1
    argent = 0.1
    gain_roi = []
    mise_totale = 0
    for i in range(n):
        mise = 1
        if test["pari gagné"][i] == 1:
                gain += mise * test["cote"][i]
                argent += mise * test.loc[i,"cote"] -1
        else:
                argent -= mise
                gain += 0
        mise_totale += mise
    ROI = (gain -  n)/ n + 1 
    return ROI

nbpari = st.sidebar.slider(label = 'Selectionnez le nombre de match sur lesquels vous voulez parier',
                      min_value = 100, max_value=28990, step=100) 

#création d'une liste de  gain en fonction du nombre de match parié
nombre_de_paris = np.linspace(1,nbpari,50,dtype=int)
gain_pari = []
ROI=[]
for i in nombre_de_paris:
    gain_pari.append(gain(i))
    ROI.append(Rent(i)) 
    
#affichage des gains en fonction du nombre de match pariés
plt.figure(figsize=(12,5))
plt.plot(nombre_de_paris, gain_pari)
plt.xlabel('nombre de paris')
plt.ylabel('gain');
plt.title(f"Gain des paris en fonction du nb de match ")
st.pyplot()


varY   = st.sidebar.slider(label = 'Ajuster la limite inférieure des ordonnées (ylim)',
                      min_value = 0.0, max_value=1.0, step=0.2) 
 
#affichage des gains en fonction du nombre de match pariés
plt.figure(figsize=(12,5))
plt.plot(nombre_de_paris, ROI)
plt.xlabel('nombre de paris')
plt.ylabel('ROI');
plt.title(f"ROI en fonction du nb de match ")
plt.ylim(varY)
st.pyplot()

