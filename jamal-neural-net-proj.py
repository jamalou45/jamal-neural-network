
# # Importation des bibliothèques utiles au projet

# In[20]:


import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split  # Import train_test_split function
import six
from IPython.display import Image
import datetime
# import pydotplus
import dash_cytoscape as cyto
import dash
from dash import dash_table
import dash_bootstrap_components as dbc
from dash import dcc, html, MATCH, ALL
from html import unescape
import dash_defer_js_import as dji
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


# # Importation de la base de données excel

credit_data = pd.ExcelFile("credit.xls")
df = pd.read_excel(credit_data, "Data", skiprows=1)
del credit_data
# on récupère une copie de du dataframe df au cas où un rechargement de la base de données est nécessaire
data = df.loc[:, df.columns[:]].copy()
del df


# - Toutes les variables sont numériques et d'après le dictionnaire, 3 d'entre elles sont qualitatives dont l'une est binaire. Nous allons dans la suite encoder les deux autres variables.
# - Pour les besoins de représentations graphiques, nous allons séparer les variables quantitatives des variables qualitatives

# In[7]:


var_quant = data.columns
var_quant = var_quant.drop(['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                            'default payment next month'])
var_quali = pd.Index(['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])

# # ENCODAGE DES VARIABLES QUALITATIVES
# - Nous allons ici encoder les variables PAY_0,.., PAY_6, EDUCATION et MARRIAGE qui présentent respectivement 11 modalités, 4 modalités et 3 modalités
# - Les nouvelles variables encodées auront le préfixe "edu" pour EDUCATION et mar_stat pour la variable MARRIAGE. le suffixe correspondra à la modalité observée.
# - La variable ID sera également supprimée
# - La variable SEX sera encodée en 0 (hommes) et 1 (femmes)

# In[8]:


data_encoded = data.copy()
data_encoded['edu_grad'] = 0
data_encoded['edu_univ'] = 0
data_encoded['edu_high'] = 0
data_encoded['mar_stat_married'] = 0
data_encoded['mar_stat_single'] = 0
data_encoded[
    [var + "_" + str(val) for var in ['PAY_' + str(i) for i in {0, 2, 3, 4, 5, 6}] for val in range(-2, 8)]] = 0
data_encoded.loc[data_encoded["SEX"] == 1, "SEX"] = 0
data_encoded.loc[data_encoded["SEX"] == 2, "SEX"] = 1

# In[9]:



# - les modalités "others" des variables EDUCATION et MARRIAGE ne sont pas encodées puisqu'elles peuvent se déduire des autres modalités encodées.

# In[10]:


data_encoded.loc[data['EDUCATION'] == 1, 'edu_grad'] = 1
data_encoded.loc[data['EDUCATION'] == 2, 'edu_univ'] = 1
data_encoded.loc[data['EDUCATION'] == 3, 'edu_high'] = 1
data_encoded.loc[data['MARRIAGE'] == 1, 'mar_stat_married'] = 1
data_encoded.loc[data['MARRIAGE'] == 2, 'mar_stat_single'] = 1

# In[11]:


for var in ['PAY_' + str(i) for i in {0, 2, 3, 4, 5, 6}]:
    for val in range(-2, 8):
        data_encoded.loc[data[var] == val, var + "_" + str(val)] = 1

# In[12]:


for ind in range(1, 7):
    data_encoded['PAY_AMT' + str(ind) + '_REL'] = data_encoded['PAY_AMT' + str(ind)] / data_encoded[
        'BILL_AMT' + str(ind)]
# on affecte le montant relatif à 0 lorsque le montant de la facture est nulle 
for ind in range(1, 7):
    data_encoded.loc[data_encoded['BILL_AMT' + str(ind)] == 0, 'PAY_AMT' + str(ind) + '_REL'] = 0

# In[13]:


data_encoded.drop(
    ['ID', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'PAY_AMT1', 'PAY_AMT2',
     'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
     'BILL_AMT6'], axis=1, inplace=True)


# - Pour les besoins de représentations graphiques, on isole les nouvelles variables qualitatives encodées obtenues
# - De même, on isole les variables quantitatives correspondant à cette nouvelle table (suppression de la variable ID)

# In[15]:


var_quali_enc = pd.Index(
    ['SEX', 'edu_grad', 'edu_univ', 'edu_high', 'mar_stat_married', 'mar_stat_single'] + [var + "_" + str(val) for var
                                                                                          in ['PAY_' + str(i) for i in
                                                                                              {0, 2, 3, 4, 5, 6}] for
                                                                                          val in range(-2, 8)])
var_quant_enc = data_encoded.columns
var_quant_enc = var_quant_enc.drop(var_quali_enc)

# - On isole ensuite la variable cible "default payment next month" puis on définit le dataframe final qui comprend tous les prédicteurs

# In[16]:


y = data_encoded['default payment next month']
data_final = data_encoded[var_quant_enc | var_quali_enc].drop(["default payment next month"], axis=1)

# In[17]:


# In[18]:


y

# # Standardisation des variables - Partitionnement de la table

# In[21]:


sscaler = preprocessing.StandardScaler().fit(data_final)
data_final_std = sscaler.transform(data_final)
x_train, x_test, y_train, y_test = train_test_split(data_final_std, y, test_size=0.3, random_state=1)

# # Modélisation par la Régression Logistique Basique


# In[24]:


# REGRESSION LOGISTIQUE BASIQUE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear', penalty='l1', max_iter=1000)

before = datetime.datetime.now()
logreg.fit(x_train, y_train)
after = datetime.datetime.now()
elap_time_log = round((after - before).total_seconds())


y_pred_log = logreg.predict(x_test)
accu_rat_log = accuracy_score(y_test, y_pred_log)

# - [Retour au sommaire](#sommaire)

# # Performances du modèle (reg-log basique)

# - Matrice de confusion

# In[25]:

confusion_matrix_log = confusion_matrix(y_test, y_pred_log)

# - Calcul de la précision et de la spécificité

# In[27]:




# - Construction de la courbe ROC

# In[28]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict_proba(x_test)[:, 1])
fpr_logit, tpr_logit, thresholds_logit = roc_curve(y_test, logreg.predict_proba(x_test)[:, 1])

# - On obtient une AUC de 0.61 sur l'échantillon test

# # Modélisation par la Régression Logistique Pénalisée

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg2 = LogisticRegression(solver='liblinear', penalty='l1', max_iter=1000)
logreg2.fit(x_train, y_train)

before = datetime.datetime.now()
logreg2.fit(x_train, y_train)
after = datetime.datetime.now()
elap_time_log2 = round((after - before).total_seconds())


y_pred_log2 = logreg2.predict(x_test)
accu_rat_log2 = accuracy_score(y_test, y_pred_log2)

# In[598]:

# - [Retour au sommaire](#sommaire)

# # Performances du modèle (reg-log pénalisée)

# In[30]:


confusion_matrix_log2 = confusion_matrix(y_test, y_pred_log2)


# - Construction de la courbe ROC

# In[33]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit2_roc_auc = roc_auc_score(y_test, logreg2.predict_proba(x_test)[:, 1])
fpr_logit2, tpr_logit2, thresholds_logit2 = roc_curve(y_test, logreg2.predict_proba(x_test)[:, 1])

# - [Retour au sommaire](#sommaire)

# # Modélisation par un arbre de décision

# In[34]:


dt = DecisionTreeClassifier(criterion='gini',
                            max_depth=10,
                            min_impurity_decrease=0.0001,
                            min_samples_leaf=0.1,
                            random_state=3)

before = datetime.datetime.now()
dt.fit(x_train, y_train)
after = datetime.datetime.now()
elap_time_tree = round((after - before).total_seconds())

y_pred_tree = dt.predict(x_test)
accu_rat_tree = accuracy_score(y_test, y_pred_tree)

confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)

# représentation de l'arbre

import datetime
from sklearn.tree import export_graphviz

now = datetime.datetime.now()
dot_data = six.StringIO()
export_graphviz(dt, out_file=dot_data, proportion=True,
                filled=True, rounded=True,
                special_characters=True, feature_names=data_final.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('modélisation avec un arbre-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.png')
Image(graph.create_png())

# In[572]:

# - Courbe ROC pour l'arbre de décision

# In[35]:


tree_roc_auc = roc_auc_score(y_test, dt.predict_proba(x_test)[:, 1])
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, dt.predict_proba(x_test)[:, 1])

# - [Retour au sommaire](#sommaire)

# # Modélisation par Random Forest

# In[36]:


# On créé un Random Forest de 100 arbres
rf = RandomForestClassifier(n_estimators=1000,
                            criterion='gini',
                            min_impurity_decrease=0.0001,
                            random_state=13,
                            max_depth=13)
# Et on lance le training sur notre dataset de train
before = datetime.datetime.now()
rf.fit(x_train, y_train)
after = datetime.datetime.now()
elap_time_rf = round((after - before).total_seconds())

y_pred_rf = rf.predict(x_test)
accu_rat_rf = accuracy_score(y_test, y_pred_rf)
# In[38]:


rf_roc_auc = roc_auc_score(y_test, rf.predict_proba(x_test)[:, 1])
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(x_test)[:, 1])


confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)


# - [Retour au sommaire](#sommaire)

# # Modélisation par XGBOOST

# In[39]:


xgb_clf = xgb.XGBClassifier(objective='binary:logistic',
                            colsample_bytree=0.4,
                            learning_rate=0.1,
                            max_depth=10,
                            alpha=19,
                            n_estimators=100,
                            random_state=13)

# In[41]:


before = datetime.datetime.now()
xgb_clf.fit(np.array(x_train), np.array(y_train))
after = datetime.datetime.now()
elap_time_xgb = round((after - before).total_seconds())

y_pred_xgb = xgb_clf.predict(np.array(x_test))
accu_rat_xgb = accuracy_score(y_test, y_pred_xgb)
confusion_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)


# In[43]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

xgb_roc_auc = roc_auc_score(np.array(y_test), xgb_clf.predict_proba(np.array(x_test))[:, 1])
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(np.array(y_test), xgb_clf.predict_proba(np.array(x_test))[:, 1])


# - [Retour au sommaire](#sommaire)

# # Modélisation par un SVM

# In[44]:


clf_svc = svm.SVC(probability=True)

before = datetime.datetime.now()
clf_svc.fit(x_train, y_train)
after = datetime.datetime.now()
elap_time_svm = round((after - before).total_seconds())

# - [Retour au sommaire](#sommaire)

# ## Performance du SVM

# In[46]:


y_pred_svm = clf_svc.predict(x_test)
accu_rat_svm = accuracy_score(y_test, y_pred_svm)

# - Matrice de confusion

# In[47]:


confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# - Calcul de la précision et de la spécificité

# In[48]:


# - Construcion de la courbe ROC

# In[53]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

svm_roc_auc = roc_auc_score(y_test, clf_svc.predict_proba(x_test)[:, 1])
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, clf_svc.predict_proba(x_test)[:, 1])

# - [Retour au sommaire](#sommaire)

# # Modélisation par un Réseau de neurones

# In[623]:

#### Fonctions disponibles dans le module sklearn
from sklearn import neural_network
# Construction du modèle
clf_model = neural_network.MLPClassifier(hidden_layer_sizes=(4, 3), activation='tanh', solver='lbfgs', max_iter=10000,
                                         alpha=1e-5, tol=1e-5)

before = datetime.datetime.now()
clf_model.fit(x_train, y_train)
after = datetime.datetime.now()
elap_time_nn = round((after - before).total_seconds())
# Prédiction sur la base de test
y_pred_nn = clf_model.predict(x_test)
accu_rat_nn = accuracy_score(y_test, y_pred_nn)


# - Construction de la courbe ROC

# In[624]:


nn_roc_auc = roc_auc_score(y_test, clf_model.predict_proba(x_test)[:, 1])
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, clf_model.predict_proba(x_test)[:, 1])

confusion_matrix_nn = confusion_matrix(y_test, y_pred_nn)

# - [Retour au sommaire](#sommaire)

# # PARTIE II : Stabilite du réseau de neurones sur la table 'kaggle : give me some credit'"

# # Importation de la base de données 'kaggle : give me some credit'

# In[57]:


var = ["SeriousDlqin2yrs",
       "RevolvingUtilizationOfUnsecuredLines",
       "Age",
       "NumberOfTime30-59DaysPastDueNotWorse",
       "DebtRatio",
       "MonthlyIncome",
       "NumberOfOpenCreditLinesAndLoans",
       "NumberOfTimes90DaysLate",
       "NumberRealEstateLoansOrLines",
       "NumberOfTimes60-89DaysPastDueNotWorse",
       "NumberOfDependents"]

# In[60]:


df_kaggle = pd.read_csv("kaggle.txt", sep=" ", names=var, header=None)

# In[61]:


data_kaggle = df_kaggle.copy()

# # Pré-traitement de la base de données

# ## - traitement des valeurs manquantes

# In[62]:


data_kaggle.isna().sum()

# - la variable présente près de 30% de valeurs manquantes, elle sera donc supprimée. Quant à la **variableNumberOfDependents**, celle-ci sera préservée et les valeurs
# manquantes correspondantes qui s'élèvent à près de 30% seront supprimées

# In[63]:


data_kaggle.drop(["MonthlyIncome"], axis=1, inplace=True)

# In[64]:


data_kaggle.drop(data_kaggle[data_kaggle["NumberOfDependents"].isna()].index, axis=0, inplace=True)
var_mod = data_kaggle.columns


# - séparation des prédicteurs de la variable cible

# In[66]:


data_final_kaggle = data_kaggle[var_mod[1:]].copy()
y_kaggle = data_kaggle[var[0]].copy()
data_final_kaggle



# ## - standardisation des variables

# In[68]:


sscaler = preprocessing.StandardScaler().fit(data_final_kaggle)
data_final_kaggle_std = sscaler.transform(data_final_kaggle)

# ## Partitionnement de la table

# In[69]:


x_train_kaggle, x_test_kaggle, y_train_kaggle, y_test_kaggle = train_test_split(data_final_kaggle_std, y_kaggle,
                                                                                test_size=0.3, random_state=1)

# # Entraînement du réseau de neurones sur la base de données

# In[70]:


before = datetime.datetime.now()
clf_model.fit(x_train_kaggle, y_train_kaggle)
after = datetime.datetime.now()
elap_time_kaggle = round((after - before).total_seconds())
# Prédiction sur la base de test
y_pred_kaggle = clf_model.predict(x_test_kaggle)
accu_rat_kaggle = accuracy_score(y_test_kaggle, y_pred_kaggle)


# # Mesure de la performance

# - Construction de la courbe ROC

# In[72]:


kaggle_roc_auc = roc_auc_score(y_test_kaggle, clf_model.predict_proba(x_test_kaggle)[:, 1])
fpr_kaggle, tpr_kaggle, thresholds_kaggle = roc_curve(y_test_kaggle, clf_model.predict_proba(x_test_kaggle)[:, 1])

confusion_matrix_kaggle = confusion_matrix(y_test_kaggle, y_pred_kaggle)

# # Comparaison avec les autres modèles du projet Big Data Analytics (Base de données 'Kaggle : Give me some credit') :

# #### Les performances des autres modèles étudiés dans le projet BDA sont dispo,nibles dans ce lien : https://houdas.shinyapps.io/dossier_bigdata_-_adamouelhajjajiselmane/
# 
# #### Notre réseau de neurones présente ici de meilleures performances en termes de pouvoir de généralisation **(AUC = 0826, accuracy ratio= 0.935)** que le modèle Random Forest du projet BDA qui présentait les meilleures performances parmi les modèles étudiés **(AUC = 0.809).**

# - [Retour au sommaire](#sommaire)




# # PARTIE III : APPLICATION DASH <a class="anchor" id="appli_dash"></a>

# In[ ]:
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME,
                                                "assets/test-image0.css", "assets/test-image.css"],
                suppress_callback_exceptions=True)
server = app.server

# In[807]:

navbar = html.Div(
    [
        html.A(html.Img(id="esa", src="assets/esa.jpg",
                        style={"border-radius": "5%", "height": "60%", "width": "100%",
                               "border-style": "solid", "border-width": "0.5vh", "border-image-outset": "0",
                               "border-color": "rgb(179, 149, 26)", "align-self": "center"}),
               href="https://www.master-esa.fr/", target="_blank",
               style={"left": "1%", "position": "absolute", "display": "flex",
                      "flex-direction": "row", "align-items": "center", "height": "17vh"}),
        html.Div(html.H2("Neural Network Project", id="titre-projet", style={"align-self": "center"}),
                 style={"display": "flex", "flex-direction": "row", "align-items": "center", "left": "20%",
                        "position": "absolute"}),

        html.A(
            [
                html.Img(id="photojam", src="assets/jamphoto.png", className="photo"),
                html.Div(id="linkedinjam", className="bi bi-linkedin", style={"color": "blue"})
            ],
            href="https://www.linkedin.com/in/jamal-el-hajjaji-data-scientist",
            target="_blank",
            id="flexjam2",
            style={"left": "50%", "position": "absolute", "display": "flex", "flex-direction": "column",
                   "align-items": "center", "height": "15vh"}),
        html.A(
            [
                html.Img(id="photobess", src="assets/bessphototrans.png", className="photo"),
                html.Div(id="linkedinbess", className="bi bi-linkedin", style={"color": "blue"})
            ],
            href="https://www.linkedin.com/in/bessem-khezami-944919154/",
            target="_blank",
            id="flexbess",
            style={"left": "58%", "position": "absolute", "display": "flex", "flex-direction": "column",
                   "align-items": "center", "height": "15vh"}),
        html.A(
            [
                html.Img(id="prof", src="assets/prof.png", className="photo", style={"background-color": "#11ffee00"}),
                html.Div(id="linkedinprof", className="bi bi-linkedin", style={"color": "blue"})
            ],
            href="http://www.leo-univ-orleans.fr/fr/membres/#abdoul-aziz.ndoye@univ-orleans.fr",
            target="_blank",
            id="flexprof",
            style={"left": "92%", "position": "absolute", "background-color": "#11ffee00", "display": "flex",
                   "flex-direction": "column", "align-items": "center", "height": "15vh"})
    ], id="particles-js",
    style={
        "position": "fixed",
        "top": 0,
        "left": "0px",
        "bottom": 0,
        "width": "100%",
        "height": "17%",
        "align-items": "center",
        "display": "flex",
        "border-width": "3px",
        "border-image-outset": "0",
        "flex-direction": "row",
        "z-index": 1,
        "background-color": "#b61924",
    }
)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "17%",
    "left": "-10px",
    "bottom": 0,
    "width": "16rem",
    "height": "100hv",
    "z-index": 1,
    "overflow": "hidden",
    "transition": "all 0.9s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
    "white-space": "pre",
    "vertical-align": "middle"
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": "17%",
    "left": "-10px",
    "bottom": 0,
    "width": "5rem",
    "height": "100hv",
    "z-index": 2,
    "overflow": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
    "text-overflow": "clip clip",
    "white-space": "pre",
    "overflow": "auto",
    "height": "800px"
}

CONTENT_STYLE = {
    "position": "fixed",
    "top": "17%",
    "transition": "margin-left .5s",
    "margin-left": "16rem",
    "margin-right": "2rem",
    "padding": "1rem 0rem",
    "background-color": "#f8f9fa",
    # "background-color": "blue",
    "overflowY": "scroll",
    "height": "83%",
    "width": "100%"
}

CONTENT_STYLE1 = {
    "position": "fixed",
    "top": "17%",
    "transition": "margin-left .9s",
    "margin-left": "5rem",
    "margin-right": "2rem",
    "padding": "1rem 0rem",
    "background-color": "#f8f9fa",
    # "background-color": "blue",
    "overflowY": "scroll",
    "height": "83%",
    "width": "100%"
}

sidebar = html.Div(
    [
        dbc.Button(outline=True, color="secondary", className="mr-1", id="btn_sidebar", style={"float": "right"},
                   children=[html.I(className="fa fa-bars", style={"color": "blue"})]),
        html.H6(html.B("Menu"), className="display-4", id="titre", style={"opacity": "1", "overflow": "hidden"}),
        html.Hr(style={"top": "50%", "margin-top": "30px", "border": "2px solid blue", "border-radius": "5px"}),
        dbc.Nav(
            [
                dbc.NavLink(className="navbtn", children=[html.I(className="bi bi-sliders", id="ico1"),
                                                          html.B("   Présentation du projet", hidden=False,
                                                                 id="navtext1", className="navtext")], href="/page-1",
                            id="page-1-link"),
                dbc.NavLink(className="navbtn", children=[html.I(className="bi bi-clipboard-data", id="ico2"),
                                                          html.B("   Données", hidden=False, id="navtext2",
                                                                 className="navtext")], href="/page-2",
                            id="page-2-link"),
                dbc.NavLink(className="navbtn", children=[html.I(className="bi bi-sliders", id="ico3"),
                                                          html.B("   Modélisation", hidden=False, id="navtext3",
                                                                 className="navtext")], href="/page-3",
                            id="page-3-link"),
                dbc.NavLink(className="navbtn", children=[html.I(className="bi bi-speedometer", id="ico4"),
                                                          html.B("   Comparaison", hidden=False, id="navtext4",
                                                                 className="navtext")], href="/page-4",
                            id="page-4-link"),
            ],
            vertical=True,
            pills=True,
            id="btn_nav"
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="page-content",
    style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id='new-nn-stored', storage_type='session'),
        dcc.Store(id='new-log2-stored', storage_type='session'),
        dcc.Store(id='new-tree-stored', storage_type='session'),
        dcc.Store(id='new-rf-stored', storage_type='session'),
        dcc.Store(id='new-xgb-stored', storage_type='session'),
        dcc.Store(id='new-svm-stored', storage_type='session'),
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        dcc.Location(id="url2"),
        navbar,
        sidebar,
        content,
        dbc.Tooltip("Jamal EL HAJJAJI", target="flexjam"),
        dbc.Tooltip("Bessem KHEZAMI", target="flexbess"),
        dbc.Tooltip("Professeur A. N'DOYE", target="flexprof"),
        dbc.Tooltip("Présentation du projet", target="page-1-link"),
        dbc.Tooltip("Données", target="page-2-link"),
        dbc.Tooltip("Modélisations", target="page-3-link"),
        dbc.Tooltip("Comparaisons", target="page-4-link"),
        dbc.Tooltip("Fonction d'activation pour les couches cachées", target="fonc-activ"),
        dbc.Tooltip("Solveur pour l'optimisation des poids", target="solveurs"),
        dbc.Tooltip("Entrez la valeur 'scale', 'auto' ou un nombre", target={"type": "params-svm", "index": "3"}),

        html.Div([html.Span(unescape('&#10006;'), className="close"),
                  html.Img(id="img01", className="modal-content"),
                  html.Div(id="caption")],
                 id="myModal", className="modal"),
        dji.Import(src="assets/particles.js"),
        dji.Import(src="assets/demo/js/app.js"),
        dji.Import(src="assets/test-image.js"),
    ],
    style={"overflow": "auto", "background-color": "red"})

data_miss = data.isna().sum().to_frame()
data_miss["variable"] = data.isna().sum().to_frame().index
data_miss["number of missing values"] = data.isna().sum().to_frame()
data_miss.index = range(1, len(data_miss) + 1)
data_miss.drop([0], axis=1, inplace=True)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
        Output("navtext1", "style"),
        Output("navtext2", "style"),
        Output("navtext3", "style"),
        Output("navtext4", "style"),
        Output("titre", "style"),
    ],
    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
            st_nav = {"display": "none", "overflow": "visible", "white-space": "pre", "text-overflow": "clip clip"}
            st_titre = {"visibility": "hidden", "overflow": "hidden", "text-overflow": "clip clip"}
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
            st_nav = {"display": "inline", "text-overflow": "clip clip"}
            st_titre = {"visibility": "visible"}
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'
        vis = True
        st_nav = {"display": "inline", "overflow": "visible"}
        st_titre = {"visibility": "visible"}

    return sidebar_style, content_style, cur_nclick, st_nav, st_nav, st_nav, st_nav, st_titre


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 5)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 5)]


obj_proj = html.Div(
    [
        html.H2(html.B("Objectifs du projet")),
        html.P("L’objectif de ce projet a consisté en la mise en oeuvre d’un réseau de neurones pour \
        modéliser la décision d’octroi de crédit. Cette modélisation  été appliquée \
        sur la base de données (“credit.xls”) de la base Kaggle"),
        html.H2(html.B("Construction du réseau")),
        html.P("Nous avons construit un réseau de neurones en choisissant l’architecture qui \
        s’adapte le mieux aux données. Nous avons ensuite vérifié la \
        robustesse du réseau construit sur les données “kaggle.txt”. Les informations relatives à \
        cette base sont disponibles dans le dictionnaire Dictionary.xls"),
        html.H2(html.B("Comparaison de modèles")),
        html.P(["Nous avons ensuite évalué la performance de notre réseau de neurones que nous avons \
        comparé aux performances des modèles suivants :", html.Br(),
                html.Span("— la regression logistique",
                          style={"display": "inline-block", "text-indent": "100px", "text-align": "left"}), html.Br(),
                html.Span(" — les arbres de décisions",
                          style={"display": "inline-block", "text-indent": "100px", "text-align": "left"}), html.Br(),
                html.Span("— les forêts aléatoires",
                          style={"display": "inline-block", "text-indent": "100px", "text-align": "left"}), html.Br(),
                html.Span("— le gradient boosting",
                          style={"display": "inline-block", "text-indent": "100px", "text-align": "left"}), html.Br(),
                html.Span("— SVM", style={"display": "inline-block", "text-indent": "100px", "text-align": "left"})])
    ], style={"text-align": "justify", "width": "100%"})

tabs_data = html.Div(
    [dcc.Tabs(id="tabs-content-inline", persistence=True, persistence_type='session', value='tab-0', children=[
        dcc.Tab(label='Aperçu de la base', value='tab-0', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Description des variables', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Statistiques descriptives', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Pré-Traitement de la table', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Données Post-Traitement', value='tab-4', style=tab_style, selected_style=tab_selected_style)
    ]),
     # html.Div(id='tabs-content-inline')
     ],
    style=tabs_styles)

tabs_models = html.Div(
    [dcc.Tabs(id="tabs-models", persistence=True, persistence_type='session', value='tab-log1', children=[
        dcc.Tab(label='Régresssion Logistique', value='tab-log1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Régresssion Logistique Pénalisée', value='tab-log2', style=tab_style,
                selected_style=tab_selected_style),
        dcc.Tab(label='Arbres de décision', value='tab-tree', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Random Forest', value='tab-rf', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Gradient Boosting', value='tab-xgb', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='SVM', value='tab-svm', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Réseaux de Neurones', value='tab-nn', style=tab_style, selected_style=tab_selected_style)
    ], colors={"background": "grey"}, style={"justify-content": "center", "justify-content": "center"})],
    style=tabs_styles)

cont_don = dcc.Loading(html.Div(id="styled-with-inline", style={"left": "-100px"}), type="default")

cont_mod = html.Div(id="styled-mod-with-inline", style={"left": "-100px"})

obj_don = html.Div(
    [
        html.H2(html.B("Présentation du jeu de données")),
        html.P(["La table credit.xls comporte des données qui s'étalent entre avril 2005 et septembre 2005 et qui portent sur les défauts de paiements de clients.\
        La table comporte ", html.B("30 000 observartions"), " décrites par ", html.B("25 variables"),
                " dont la signification et la description statistique sont données ci-dessous."]),
        tabs_data, cont_don
    ],
    style={"text-align": "justify", "width": "85%", "overflow": "auto", "height": "800px%"})

toast_params_log2 = dbc.Toast("Veuillez renseigner tous les paramètres", id="alert-params-log2", header="Avertissement",
                              is_open=False, dismissable=True, icon="danger",
                              style={"position": "fixed", "top": "50%", "right": "40vw", "height": "10vh",
                                     "width": "20vw"})
toast_calcul_log2 = dbc.Toast("Calcul en cours...veuillez rester sur cette page", id="alert-calcul-log2",
                              header="Information", is_open=False, dismissable=True, icon="danger",
                              style={"position": "fixed", "top": "60%", "right": "40vw", "height": "10vh",
                                     "width": "20vw"})

toast_params_tree = dbc.Toast("Veuillez renseigner tous les paramètres", id="alert-params-tree", header="Avertissement",
                              is_open=False, dismissable=True, icon="danger",
                              style={"position": "fixed", "top": "50%", "right": "40vw", "height": "10vh",
                                     "width": "20vw"})
toast_calcul_tree = dbc.Toast("Calcul en cours...veuillez rester sur cette page", id="alert-calcul-tree",
                              header="Information", is_open=False, dismissable=True, icon="danger",
                              style={"position": "fixed", "top": "60%", "right": "40vw", "height": "10vh",
                                     "width": "20vw"})

toast_calcul_rf = dbc.Toast("Calcul en cours...veuillez rester sur cette page", id="alert-calcul-rf",
                            header="Information", is_open=False, dismissable=True, icon="danger",
                            style={"position": "fixed", "top": "60%", "right": "40vw", "height": "10vh",
                                   "width": "20vw"})

toast_params_xgb = dbc.Toast("Veuillez renseigner tous les paramètres", id="alert-params-xgb", header="Avertissement",
                             is_open=False, dismissable=True, icon="danger",
                             style={"position": "fixed", "top": "50%", "right": "40vw", "height": "10vh",
                                    "width": "20vw"})
toast_calcul_xgb = dbc.Toast("Calcul en cours...veuillez rester sur cette page", id="alert-calcul-xgb",
                             header="Information", is_open=False, dismissable=True, icon="danger",
                             style={"position": "fixed", "top": "60%", "right": "40vw", "height": "10vh",
                                    "width": "20vw"})

toast_params_svm = dbc.Toast("Veuillez renseigner tous les paramètres", id="alert-params-svm", header="Avertissement",
                             is_open=False, dismissable=True, icon="danger",
                             style={"position": "fixed", "top": "50%", "right": "40vw", "height": "10vh",
                                    "width": "20vw"})
toast_calcul_svm = dbc.Toast("Calcul en cours...veuillez rester sur cette page", id="alert-calcul-svm",
                             header="Information", is_open=False, dismissable=True, icon="danger",
                             style={"position": "fixed", "top": "60%", "right": "40vw", "height": "10vh",
                                    "width": "20vw"})

toast_params_nn = dbc.Toast("Veuillez renseigner tous les paramètres", id="alert-params-nn", header="Avertissement",
                            is_open=False, dismissable=True, icon="danger",
                            style={"position": "fixed", "top": "50%", "right": "40vw", "height": "10vh",
                                   "width": "20vw"})
toast_calcul_nn = dbc.Toast("Calcul en cours...veuillez rester sur cette page", id="alert-calcul-nn",
                            header="Information", is_open=False, dismissable=True, icon="danger",
                            style={"position": "fixed", "top": "60%", "right": "40vw", "height": "10vh",
                                   "width": "20vw"})

obj_mod = html.Div(
    [
        html.H2(html.B("Modélisations à l'aide de différents algorithmes")),
        html.P(["Nous présentons ici les modèles prédictifs utilisés, les paramètres optimaux associés retenus ainsi que leur performance respective.\
        Nous donnons également à l'internaute la possibilité de tester d'autres valeurs pour quelques hyperparamètres seulement \
        afin de conserver une navigation fluide. Le partitionnement utilisé pour les phases d'entraînement et de test est de 70 - 30"]),
        tabs_models, cont_mod,
        toast_params_log2, toast_calcul_log2,
        toast_params_tree, toast_calcul_tree,
        toast_calcul_rf,
        toast_params_xgb, toast_calcul_xgb,
        toast_params_svm, toast_calcul_svm,
        toast_params_nn, toast_calcul_nn,
    ],
    style={"text-align": "justify", "width": "85%", "overflow": "auto", "height": "1000px%"})

table_header = [html.Thead(html.Tr([html.Th("Variable", style={"width": "25%"}), html.Th("Description")]),
                           style={"position": "sticky", "top": "0"})]

row1 = html.Tr([html.Td("ID"), html.Td("ID of the client")])
row2 = html.Tr([html.Td("LIMIT_BAL"), html.Td(
    "Amount of the given credit (NT dollar). It includes both the individual consumer credit and his/her family (supplementary) credit")])
row3 = html.Tr([html.Td("SEX"), html.Td("Gender (1 = male; 2 = female)")])
row4 = html.Tr([html.Td("EDUCATION"),
                html.Td("Education level (1 = graduate school; 2 = university; 3 = high school; 4 = others")])
row5 = html.Tr([html.Td("MARRIAGE"), html.Td("Marital status (1 = married; 2 = single; 3 = others)")])
row6 = html.Tr([html.Td("AGE"), html.Td("...in years")])
row7 = html.Tr([html.Td("PAY_0"), html.Td(
    "Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)")])
row8 = html.Tr([html.Td("PAY_2"), html.Td("Repayment status in August, 2005")])
row9 = html.Tr([html.Td("PAY_3"), html.Td("Repayment status in July, 2005")])
row10 = html.Tr([html.Td("PAY_4"), html.Td("Repayment status in June, 2005")])
row11 = html.Tr([html.Td("PAY_5"), html.Td("Repayment status in May, 2005")])
row12 = html.Tr([html.Td("PAY_6"), html.Td("Repayment status in April, 2005")])
row13 = html.Tr([html.Td("BILL_AMT1"), html.Td("Amount of bill statement in September, 2005 (NT dollar)")])
row14 = html.Tr([html.Td("BILL_AMT2"), html.Td("Amount of bill statement in August, 2005 (NT dollar)")])
row15 = html.Tr([html.Td("BILL_AMT3"), html.Td("Amount of bill statement in July, 2005 (NT dollar)")])
row16 = html.Tr([html.Td("BILL_AMT4"), html.Td("Amount of bill statement in June, 2005 (NT dollar)")])
row17 = html.Tr([html.Td("BILL_AMT5"), html.Td("Amount of bill statement in May, 2005 (NT dollar)")])
row18 = html.Tr([html.Td("BILL_AMT6"), html.Td("Amount of bill statement in April, 2005 (NT dollar)")])
row19 = html.Tr([html.Td("PAY_AMT1"), html.Td("Amount of previous payment in September, 2005 (NT dollar)")])
row20 = html.Tr([html.Td("PAY_AMT2"), html.Td("Amount of previous payment in August, 2005 (NT dollar)")])
row21 = html.Tr([html.Td("PAY_AMT3"), html.Td("Amount of previous payment in July, 2005 (NT dollar)")])
row22 = html.Tr([html.Td("PAY_AMT4"), html.Td("Amount of previous payment in June, 2005 (NT dollar)")])
row23 = html.Tr([html.Td("PAY_AMT5"), html.Td("Amount of previous payment in May, 2005 (NT dollar)")])
row24 = html.Tr([html.Td("PAY_AMT6"), html.Td("Amount of previous payment in April, 2005 (NT dollar)")])
row25 = html.Tr(
    [html.Td("default payment next month"), html.Td("Default payment (Yes = 1, No = 0), as the response variable. ")])

table_body = [html.Tbody(
    [row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15, row16, row17,
     row18, row19, row20, row21, row22, row23, row24, row25]
    )]

table = html.Div(dbc.Table(table_header + table_body,
                           bordered=True,
                           hover=True,
                           dark=True,
                           striped=True),
                 style={"overflow": "hidden auto", "max-height": "370px"}
                 )

fig_comp = go.Figure()
fig_comp.data = []
fig_comp.layout = {}
fig_comp.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig_comp.add_trace(
    go.Scatter(x=fpr_logit, y=tpr_logit, name="régression logistique simple, AUC = {:.3f}".format(logit_roc_auc),
               mode='lines'))
fig_comp.add_trace(
    go.Scatter(x=fpr_logit2, y=tpr_logit2, name="régression logistique pénalisée, AUC = {:.3f}".format(logit2_roc_auc),
               mode='lines'))
fig_comp.add_trace(
    go.Scatter(x=fpr_tree, y=tpr_tree, name="Arbre de décision, AUC = {:.3f}".format(tree_roc_auc), mode='lines'))
fig_comp.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name="Random Forest, AUC = {:.3f}".format(rf_roc_auc), mode='lines'))
fig_comp.add_trace(
    go.Scatter(x=fpr_xgb, y=tpr_xgb, name="Gradient Boosting, AUC = {:.3f}".format(xgb_roc_auc), mode='lines'))
fig_comp.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, name="SVM, AUC = {:.3f}".format(svm_roc_auc), mode='lines'))
fig_comp.add_trace(
    go.Scatter(x=fpr_nn, y=tpr_nn, name="Réseau de neurones, AUC = {:.3f}".format(nn_roc_auc), mode='lines'))
fig_comp.update_layout(width=700, height=400, font=dict(size=15), legend=dict(x=0.35, y=0.05, bgcolor="LightSteelBlue"),
                       margin=dict(l=20, r=10, t=10, b=20))

obj_comp = html.Div(
    [
        html.H2(
            html.B("Comparaison des pouvoirs prédictifs de chaque modèle appliqué sur la base de données credit.xls")),
        html.P("Nous rassemblons ici les résultats obtenus pour chaque modèle"),
        html.Br(),
        html.Div([
            html.Div([html.H3(html.B("Résultats"), style={"text-align": "center"}),
                      html.Div(dbc.Table(
                          [html.Thead(html.Tr([html.Th("Paramètres", style={"text-align": "center", "width": "30%"}),
                                               html.Th("AUC", style={"text-align": "center"}),
                                               html.Th("Accuracy Ratio", style={"text-align": "center"}),
                                               html.Th("Temps écoulé (secondes)", style={"text-align": "center"})],
                                              style={"width": "25%"}),
                                      style={"position": "sticky", "top": "0"})] + \
                          [html.Tbody([html.Tr(
                              [html.Td("Régression Logistique simple"), html.Td("{:.3f}".format(logit_roc_auc)),
                               html.Td("{:.3f}".format(accu_rat_log)), html.Td(elap_time_log)]),
                                       html.Tr([html.Td("Régression Logistique pénalisée"),
                                                html.Td("{:.3f}".format(logit2_roc_auc)),
                                                html.Td("{:.3f}".format(accu_rat_log2)), html.Td(elap_time_log2)]),
                                       html.Tr([html.Td("Arbre de décision"), html.Td("{:.3f}".format(tree_roc_auc)),
                                                html.Td("{:.3f}".format(accu_rat_tree)), html.Td(elap_time_tree)]),
                                       html.Tr([html.Td("Random Forest"), html.Td("{:.3f}".format(rf_roc_auc)),
                                                html.Td("{:.3f}".format(accu_rat_rf)), html.Td(elap_time_rf)]),
                                       html.Tr([html.Td("Gradient Boosting"), html.Td("{:.3f}".format(xgb_roc_auc)),
                                                html.Td("{:.3f}".format(accu_rat_xgb)), html.Td(elap_time_xgb)]),
                                       html.Tr([html.Td("SVM"), html.Td("{:.3f}".format(svm_roc_auc)),
                                                html.Td("{:.3f}".format(accu_rat_svm)), html.Td(elap_time_svm)]),
                                       html.Tr([html.Td("Réseau de Neurones"), html.Td("{:.3f}".format(nn_roc_auc)),
                                                html.Td("{:.3f}".format(accu_rat_nn)), html.Td(elap_time_nn)])
                                       ])],
                          bordered=True,
                          hover=True,
                          dark=True,
                          striped=True, style={"overflow": "auto", "width": "100%"}),
                               style={"overflow": "auto", "height": "400px"}, className="table-wrapper")],
                     style={"width": "40%", "align-self": "center", "margin": "20px"}),
            html.Div([html.H3(html.B("Courbe ROC - AUC"), style={"text-align": "center"}),
                      html.Br(),
                      html.Div(dcc.Graph(figure=fig_comp), style={"overflow": "auto"})
                      ],
                     style={"height": "100%", "width": "60%", "display": "flex", "flex-direction": "column",
                            "align-items": "center"})
        ], style={"display": "flex", "flex-direction": "row", "align-items": "center"}),
        html.H2(html.B("Conclusion")),
        html.P("Les résultats obtenus nous ont permis d'identifier le modèle XGBoost comme étant le modèle le plus performant aussi bien en termes de pouvoir prédictif qu'en termes de temps de calcul.\
                   Le modèle SVM est le moins performant des modèles utilisé avec une AUC de 0.684. Le réseau de neurones est classé cinquième. En revanche, notons que le réseau de neurones retenu est de loin le plus performant lorsqu'il est appliqué sur la table \
                   'Kaggle : Give me some credit'")
    ], style={"width": "90%"})


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/page-1"]:
        return obj_proj
    elif pathname == "/page-2":
        return obj_don
    elif pathname == "/page-3":
        return obj_mod
    elif pathname == "/page-4":
        return obj_comp
    else:
        return obj_proj
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


obj_table = dcc.Loading(dash_table.DataTable(
    page_action="native",
    data=data.to_dict('records'),
    columns=[{"name": c, "id": c, 'selectable': True} for c in data.columns],
    style_data={
        'backgroundColor': 'rgb(30,30,30)',
        'textAlign': 'center',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(50, 50, 50)'},
        {
            'if': {'column_id': 'default payment next month'},
            'backgroundColor': 'white', 'color': 'blue',
            'fontWeight': 'Bold', 'width': '260px',
        },
        {
            'if': {'column_id': 'ID'},
            'fontWeight': 'Bold', 'backgroundColor': 'rgb(100,100,100)'

        }],
    page_size=50,
    fixed_rows={'headers': True, 'data': 0},
    fixed_columns={'headers': True, 'data': 1},
    style_table={"maxWidth": "100%", "height": "500px", "overflow": "auto"},
    style_header={'fontWeight': 'Bold', 'backgroundColor': 'rgb(100,100,100)'},
    style_header_conditional=[
        {
            'if': {'column_id': 'default payment next month'},
            'width': '350px', 'color': 'blue'
        },
        {
            'if': {'column_id': 'ID'},
            'backgroundColor': 'rgb(150,150,150)'
        }
    ],
    style_cell={
        'color': 'white',
        'textAlign': 'center',
        'minWidth': '100px',
        'maxWidth': '800px',
    }
), type="default")


@app.callback(Output('styled-with-inline', 'children'),
              Input('tabs-content-inline', 'value'))
def render_content_tab(tab):
    if tab == "tab-0":
        return obj_table
    if tab == 'tab-1':
        return [table,
                "Toutes les variables sont quantitatives à l'exception des variables catégorielles suivantes qui seront encodées ultérieurement à l'étape du pré-traitement :",
                html.Br(),
                html.B(["ID", html.Br(),
                        "    SEX", html.Br(),
                        "    EDUCATION", html.Br(),
                        "    MARRIAGE", html.Br(),
                        "    PAY_0", html.Br(),
                        "    PAY_2", html.Br(),
                        "    PAY_3", html.Br(),
                        "    PAY_4", html.Br(),
                        "    PAY_5", html.Br(),
                        "    PAY_6"])]
    elif tab == 'tab-2':
        return [dcc.Dropdown(data.columns, id='demo-dropdown', value=data.columns[-1], persistence=True,
                             persistence_type='session'),
                dcc.Graph(id='graph2'),
                dcc.Graph(id='graph3')]
    elif tab == 'tab-3':
        return [html.Br(),
                html.Div([
                    html.Div([
                        html.H3(html.B("Traitement des valeurs manquantes"), style={"text-align": "center"}),
                        html.Br(),
                        html.Div(dbc.Table.from_dataframe(data_miss, striped=True, bordered=True, hover=True),
                                 className="table-wrapper",
                                 style={"overflow": "auto", "width": "80%", "height": "500px"})],
                        style={"display": "flex", "align-items": "center", "flex-direction": "column", "width": "35%",
                               "margin": "10px"}),
                    html.Div([html.H3(html.B("Matrice des corrélations"), style={"text-align": "center"}), html.Br(),
                              html.Div(dcc.Graph(id='graph4', figure=trace),
                                       style={"width": "80%", "overflow": "auto", "height": "500px"})],
                             style={"display": "flex", "align-items": "center", "flex-direction": "column",
                                    "margin": "10px", "align-self": "center", "width": "65%"})],
                    style={"width": "100%", "display": "flex", "align-items": "start", "flex-direction": "row"}),
                html.P("- Nous constatons que la table ne présente aucune valeur manquante"),
                html.P(
                    "- Le taux de défaut observé qui est de 22% montre que la base de données n'a pas besoin d'être rééchantillonée."),
                html.P(["- La matrice montre des corrélations élevées entre les variables BILL_AMT1,...,BIL_AMT6 qui, pour rappel,\
        représentent les montants des factures à rembourser pour chaque mois d'avril 2005 à septembre 2005. Ainsi, pour représenter plus fidèlement les montants \
        remboursés par les clients, nous décidons dans la suite de construire des variables relatives en rapportant les montants remboursés PAY_AMT1,...,PAY_AMT6 \
        aux montants des factures associés BILL_AMT1,...,BIL_AMT6 : ", html.B("PAY_AMT_REL1 = PAY_AMT1 / BILL_AMT1"),
                        " (pour le mois de Septembre 2005)"]),
                html.P(["- La variable ", html.B("ID"),
                        " servant simplement à identifier les clients est supprimée dans la suite"]),
                html.H3(html.B("Encodage des variables catégorielles"), id="encod"),
                html.P(["- Nous encodons les variables SEX, EDUCATION et MARRIAGE", html.Br(),
                        "- Nous encodons également les variables catégorielles PAY_0, PAY_2, PAY_3, PAY_4, PAY_5 et PAY_6 qui comprennent les 11 modalités suivantes {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}",
                        html.Br(),
                        "Après encodage, la variable PAY_0 génerera par exemple 10 variables ", html.B("PAY_0_-2, PAY_0_-1, PAY_0_0, PAY_0_1, PAY_0_2, PAY_0_3, PAY_0_4, PAY_0_5, PAY_0_6,\
               PAY_0_7"),
                        html.Br(),
                        "A l'issue de l'encodage, nous nous retrouvenons avec 74 variables explicatives"])]
    elif tab == 'tab-4':
        return [html.Br(),
                html.H3(html.B("Statistiques descriptives")),
                dcc.Dropdown(data_encoded.columns, id='demo-dropdown2', value=data_encoded.columns[0], persistence=True,
                             persistence_type='session'),
                html.Div([
                    html.Div(dcc.Graph(id='graph5'), style={"width": "50%"}),
                    html.Div(dcc.Graph(id='graph6'), style={"width": "50%"})],
                    style={"margin": "20px", "display": "flex"}),
                html.H3(html.B("Matrice des corrélations")),
                dcc.Graph(id='graph7', figure=trace_enc),
                html.P(
                    ["- Nous constatons que les corrélations sur les nouvelles variables relatives crées ont disparu",
                     html.Br(),
                     "- Les valeurs NaN observées dans la matrice sont dues au fait que les variables correspondantes (PAY_5_1 et PAY_6_1) prennent une valeur nulle",
                     html.Br(),
                     "Autrement dit, un retard de paiement d'un mois n'a jamais été observé pour le mois de mai 2005"])
                ]
    else:
        return obj_table


corr = data.corr()
trace = go.Figure(layout=go.Layout(height=700, margin=dict(l=20, r=20, t=40, b=20))).add_trace(go.Heatmap(z=corr.values,
                                                                                                          x=corr.index.values,
                                                                                                          y=corr.columns.values))

corr_enc = data_encoded.corr()
trace_enc = go.Figure(layout=go.Layout(height=700, margin=dict(l=20, r=20, t=40, b=20))).add_trace(
    go.Heatmap(z=corr_enc.values,
               x=corr_enc.index.values,
               y=corr_enc.columns.values))


@app.callback([Output('graph2', component_property='figure'), Output('graph3', component_property='figure')],
              Input('demo-dropdown', component_property='value'))
def update_figure(value):
    if value in var_quant and value != 'default payment next month':
        return px.histogram(data, x=value, color='default payment next month').update_layout(
            legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.95)), go.Figure(
            layout=go.Layout(height=700, margin=dict(l=20, r=20, t=40, b=20))).add_trace(go.Box(
            y=data[value],
            name=value,
            marker_color='darkblue',
            boxmean=True))
    else:
        return px.histogram(data, x=value, barmode='group', color='default payment next month'), px.pie(data,
                                                                                                        names=value,
                                                                                                        title='Distribution of ' + value,
                                                                                                        hole=.7)


my_stylesheet = [
    {
        'selector': '.node_par',
        'style': {'content': 'data(label)'}
    },
    {
        'selector': '.node_chil',
        'style': {'content': 'data(label)', 'text-halign': 'center', 'text-valign': 'center', 'width': 'label',
                  'height': 'label'}
    },
    {
        'selector': '.green',
        'style': {
            'background-color': 'green'
        }
    },
    {
        'selector': '.red',
        'style': {
            'background-color': 'red',
            'line-color': 'red'
        }
    },
    {
        'selector': '.blue',
        'style': {
            'background-color': 'blue'
        }
    }
]

neur_net = cyto.Cytoscape(id='cytoscape-compound',
                          layout={'name': 'preset'},
                          stylesheet=my_stylesheet,
                          style={'width': '100%', 'height': '50vh'},
                          elements=[],
                          autoRefreshLayout=True
                          )


@app.callback([Output('graph5', component_property='figure'), Output('graph6', component_property='figure')],
              Input('demo-dropdown2', component_property='value'))
def update_figure(value):
    if value in var_quant_enc and value != 'default payment next month':
        return px.histogram(data_encoded, x=value, color='default payment next month').update_layout(
            legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.95)), go.Figure(
            layout=go.Layout(margin=dict(l=20, r=20, t=40, b=20))).add_trace(go.Box(
            y=data_encoded[value],
            name=value,
            marker_color='darkblue',
            boxmean=True))
    else:
        return px.histogram(data_encoded, x=value, barnorm='percent', barmode='group',
                            color='default payment next month'), px.pie(data_encoded, names=value,
                                                                        title='Distribution of ' + value, hole=.7)


@app.callback(Output('styled-mod-with-inline', 'children'),
              Input('tabs-models', 'value'))
def render_model_content(tab):
    if tab == 'tab-log1':
        conf_mat_logit = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_log), index=True, striped=True,
                                                  hover=True, bordered=True, style={"width": "100%"})
        conf_mat_logit.children[0].children[0].children[0].children = ""
        conf_mat_logit.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_logit.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.H3(html.B("Performances du modèle")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_logit,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_log)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(logit_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_log))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div(
                          [html.H3(html.B("Courbe ROC - Basic Log Reg"), style={"text-align": "center"}), html.Br(),
                           dcc.Graph(
                               figure=px.area(x=fpr_logit, y=tpr_logit, title=f'ROC Curve (AUC={logit_roc_auc:.3f})',
                                              labels=dict(x='False Positive Rate', y='True Positive Rate')).add_shape(
                                   type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                   margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                               style={"margin-bottom": "10px", "width": "80%"})
                           ],
                          style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                 "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br()
        ])
    elif tab == 'tab-log2':
        conf_mat_logit2 = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_log2), index=True, striped=True,
                                                   hover=True, bordered=True, style={"width": "100%"})
        conf_mat_logit2.children[0].children[0].children[0].children = ""
        conf_mat_logit2.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_logit2.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.H3(html.B("Paramètres optimaux")),
            dbc.Table([html.Thead(html.Tr([html.Th("Paramètres", style={"width": "50%"}), html.Th("Valeur")]),
                                  style={"position": "sticky", "top": "0"})] + \
                      [html.Tbody([html.Tr([html.Td("Solveur utilisé pour l'optimisation"), html.Td("liblinear")]),
                                   html.Tr([html.Td("Norme utilisée pour la pénalisation"), html.Td("L1")]),
                                   html.Tr([html.Td("Coefficient C : inverse de l'intensité de la pénalisation"),
                                            html.Td("1")]),
                                   html.Tr([html.Td("Tolérance pour le critère d'arrêt tol"), html.Td("0.0001")])])],
                      bordered=True,
                      hover=True,
                      dark=True,
                      striped=True, style={"width": "30%"}),
            dcc.Link(html.I("Détails sur les paramètres du modèle LogisticRegression"),
                     href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
                     target="_blank"),
            html.Br(),
            html.H3(html.B("Performances du modèle")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_logit2,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_log2)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(logit2_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_log2))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div(
                          [html.H3(html.B("Courbe ROC - Penalized Log Reg"), style={"text-align": "center"}), html.Br(),
                           dcc.Graph(
                               figure=px.area(x=fpr_logit2, y=tpr_logit2, title=f'ROC Curve (AUC={logit2_roc_auc:.3f})',
                                              labels=dict(x='False Positive Rate', y='True Positive Rate')).add_shape(
                                   type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                   margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                               style={"margin-bottom": "10px", "width": "80%"})
                           ],
                          style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                 "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Test de nouveaux paramètres")),
            html.Div([
                html.Div([html.P("Solveur", style={"text-align": "center"}),
                          dcc.Dropdown(["lbfgs", "newton-cg", "liblinear", "sag", "saga"], value="liblinear",
                                       clearable=False, persistence=True, persistence_type="session",
                                       id={"type": "params-log2", "index": "1"})],
                         style={"margin": "0 20px 0 20px", "width": "8vh"}),
                html.Div([html.P("Terme de pénalisation"),
                          dcc.Dropdown(["none", "l1", "l2", "elasticnet"], value="l1", clearable=False,
                                       persistence=True, persistence_type="session",
                                       id={"type": "params-log2", "index": "2"})],
                         style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Coefficient C"), dbc.Input(placeholder="Renseigner une valeur...", type="number",
                                                             id={"type": "params-log2", "index": "3"})],
                         style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Tolérance tol"), dbc.Input(placeholder="Renseigner une valeur...", type="number",
                                                             id={"type": "params-log2", "index": "4"})],
                         style={"margin": "0 20px 0 20px"}),
                dcc.Loading(dbc.Button("Lancez le calcul", id="btn-calc-log2", style={"height": "80%"}))
            ], style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Nouveaux résultats obtenus pour le régression logistique pénalisée")),
            html.Br(),
            dcc.Loading(html.Div(id="log2-new-result",
                                 style={"border": "solid", "display": "flex", "align-items": "center",
                                        "flex-direction": "row"}))
        ])
    elif tab == 'tab-tree':
        conf_mat_tree = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_tree), index=True, striped=True,
                                                 hover=True, bordered=True, style={"width": "100%"})
        conf_mat_tree.children[0].children[0].children[0].children = ""
        conf_mat_tree.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_tree.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.Div([
                html.Div([html.H3(html.B("Paramètres optimaux")),
                          dbc.Table([html.Thead(
                              html.Tr([html.Th("Paramètres", style={"width": "70%"}), html.Th("Valeur")]),
                              style={"position": "sticky", "top": "0"})] + \
                                    [html.Tbody([html.Tr([html.Td("Critère d'arrêt"), html.Td("Indice de Gini")]),
                                                 html.Tr([html.Td("Décroissance minimale de l'indice de Gini"),
                                                          html.Td("0.0001")]),
                                                 html.Tr([html.Td("Proportion minimale d'échantillon par feuille "),
                                                          html.Td("0.1")]),
                                                 html.Tr([html.Td("Profondeur maximale"), html.Td("10")])])],
                                    bordered=True,
                                    hover=True,
                                    dark=True,
                                    striped=True, style={"width": "50%", "align-self": "center"}),
                          dcc.Link(html.I("Détails sur les paramètres du modèle DecisionTreeClassifier"),
                                   href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
                                  target="_blank")],
                         style={"display": "flex", "flex-direction": "column", "align-items": "center",
                                "width": "50%"}),
                html.Div([html.H3(html.B("Architecture de l'arbre optimal"), style={"text-align": "center"}),
                          html.Div(html.Img(src="assets/modélisation avec un arbre de décision.png", id="myImg",
                                            alt="Arbre de décision",
                                            style={"width": "100%"}),
                                   style={"border": "solid", "overflow": "auto", "height": "400px"})],
                         style={"display": "flex", "flex-direction": "column", "width": "50%"})],
                style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Performances du modèle")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_tree,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_tree)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(tree_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_tree))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div([html.H3(html.B("Courbe ROC - Tree"), style={"text-align": "center"}), html.Br(),
                                dcc.Graph(
                                    figure=px.area(x=fpr_tree, y=tpr_tree, title=f'ROC Curve (AUC={tree_roc_auc:.3})',
                                                   labels=dict(x='False Positive Rate',
                                                               y='True Positive Rate')).add_shape(
                                        type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                        margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                    style={"margin-bottom": "10px", "width": "80%"})
                                ],
                               style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Test de nouveaux paramètres")),
            html.Div([
                html.Div([html.P("Profondeur maximale"),
                          dcc.Dropdown([i for i in range(1, 100)], id={"type": "params-tree", "index": "1"},
                                       clearable=False, persistence=True,
                                       persistence_type="session", value=10)], style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Proportion minimale par feuille"),
                          dbc.Input(placeholder="Renseigner une valeur...", type="number",
                                    id={"type": "params-tree", "index": "2"})], style={"margin": "0 20px 0 20px"}),
                dcc.Loading(dbc.Button("Lancez le calcul", id="btn-calc-tree", style={"height": "80%"}))
            ], style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Nouveaux résultats obtenus pour l'arbre de décision")),
            html.Br(),
            dcc.Loading(html.Div(id="tree-new-result",
                                 style={"border": "solid", "display": "flex", "align-items": "center",
                                        "flex-direction": "row"}))
        ])
    elif tab == 'tab-rf':
        conf_mat_rf = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_rf), index=True, striped=True, hover=True,
                                               bordered=True, style={"width": "100%"})
        conf_mat_rf.children[0].children[0].children[0].children = ""
        conf_mat_rf.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_rf.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.H3(html.B("Paramètres optimaux")),
            dbc.Table([html.Thead(html.Tr([html.Th("Paramètres", style={"width": "60%"}), html.Th("Valeur")]),
                                  style={"position": "sticky", "top": "0"})] + \
                      [html.Tbody([html.Tr([html.Td("Nombre d'arbres"), html.Td("1000")]),
                                   html.Tr([html.Td("Critère d'homogénéité"), html.Td("Indice de Gini")]),
                                   html.Tr([html.Td("Décroissance minimale de l'impureté"), html.Td("0.0001")]),
                                   html.Tr([html.Td("Profondeur maximale"), html.Td("13")])])],
                      bordered=True,
                      hover=True,
                      dark=True,
                      striped=True, style={"width": "25%"}),
            dcc.Link(html.I("Détails sur les paramètres du modèle RandomForestClassifier"),
                     href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
                     target="_blank"),
            html.Br(),
            html.H3(html.B("Performances du modèle")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_rf,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_rf)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(rf_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_rf))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div([html.H3(html.B("Courbe ROC - RF"), style={"text-align": "center"}), html.Br(),
                                dcc.Graph(figure=px.area(x=fpr_rf, y=tpr_rf, title=f'ROC Curve (AUC={rf_roc_auc:.3f})',
                                                         labels=dict(x='False Positive Rate',
                                                                     y='True Positive Rate')).add_shape(
                                    type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                    margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                          style={"margin-bottom": "10px", "width": "80%"})
                                ],
                               style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Test de nouveaux paramètres")),
            html.Div([
                html.Div([html.P("Nombre d'arbres"),
                          dcc.Dropdown([j * 10 ** i for i in range(1, 5) for j in range(1, 10)] + [10000],
                                       id={"type": "params-rf", "index": "1"}, clearable=False, persistence=True,
                                       persistence_type="session", value=10)], style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Profondeur maximale"),
                          dcc.Dropdown([i for i in range(1, 50)], id={"type": "params-rf", "index": "2"},
                                       clearable=False, persistence=True,
                                       persistence_type="session", value=10)], style={"margin": "0 20px 0 20px"}),
                dcc.Loading(dbc.Button("Lancez le calcul", id="btn-calc-rf", style={"height": "80%"}))
            ], style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Nouveaux résultats obtenus pour le Random Forest")),
            html.Br(),
            dcc.Loading(html.Div(id="rf-new-result",
                                 style={"border": "solid", "display": "flex", "align-items": "center",
                                        "flex-direction": "row"}))
        ])
    elif tab == 'tab-xgb':
        conf_mat_xgb = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_xgb), index=True, striped=True,
                                                hover=True, bordered=True, style={"width": "100%"})
        conf_mat_xgb.children[0].children[0].children[0].children = ""
        conf_mat_xgb.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_xgb.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.H3(html.B("Paramètres optimaux")),
            dbc.Table([html.Thead(html.Tr([html.Th("Paramètres", style={"width": "50%"}), html.Th("Valeur")]),
                                  style={"position": "sticky", "top": "0"})] + \
                      [html.Tbody([html.Tr([html.Td("Objective"), html.Td(
                          "binary:logistic, régression logistique pour la classification binaire")]),
                                   html.Tr([html.Td(
                                       "Taux d'échantillonage des variables pour chaque nouvel arbre construit :  colsample_bytree"),
                                            html.Td("0.4")]),
                                   html.Tr([html.Td("Taux d'apprentissage: learning_rate"), html.Td("0.1")]),
                                   html.Tr([html.Td("Profondeur maximale: max_depth"), html.Td("10")]),
                                   html.Tr([html.Td("Poids utilisé pour la pénalisation L1: alpha"), html.Td("19")]),
                                   html.Tr([html.Td("Nombre d'arbres"), html.Td("100")])])],
                      bordered=True,
                      hover=True,
                      dark=True,
                      striped=True, style={"width": "50%"}),
            dcc.Link(html.I("Détails sur les paramètres du modèle XGB"),
                     href="https://xgboost.readthedocs.io/en/stable/python/python_api.html", target="_blank"),
            html.Br(),
            html.H3(html.B("Performances du modèle")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_xgb,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_xgb)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(xgb_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_xgb))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div([html.H3(html.B("Courbe ROC - XGB"), style={"text-align": "center"}), html.Br(),
                                dcc.Graph(
                                    figure=px.area(x=fpr_xgb, y=tpr_xgb, title=f'ROC Curve (AUC={xgb_roc_auc:.3f})',
                                                   labels=dict(x='False Positive Rate',
                                                               y='True Positive Rate')).add_shape(
                                        type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                        margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                    style={"margin-bottom": "10px", "width": "80%"})
                                ],
                               style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Test de nouveaux paramètres")),
            html.Div([
                html.Div([html.P("Nombre d'arbres"),
                          dcc.Dropdown([j * 10 ** i for i in range(1, 5) for j in range(1, 10)] + [10000],
                                       id={"type": "params-xgb", "index": "1"}, clearable=False, persistence=True,
                                       persistence_type="session", value=10)], style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Profondeur maximale"),
                          dcc.Dropdown([i for i in range(1, 50)], id={"type": "params-xgb", "index": "2"},
                                       clearable=False, persistence=True,
                                       persistence_type="session", value=10)], style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("colsample_bytree"), dbc.Input(placeholder="Renseigner une valeur...", type="number",
                                                                id={"type": "params-xgb", "index": "3"})],
                         style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Learning rate"), dbc.Input(placeholder="Renseigner une valeur...", type="number",
                                                             id={"type": "params-xgb", "index": "4"})],
                         style={"margin": "0 20px 0 20px"}),
                dcc.Loading(dbc.Button("Lancez le calcul", id="btn-calc-xgb", style={"height": "80%"}))
            ], style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Nouveaux résultats obtenus pour le Gradient Boosting")),
            html.Br(),
            dcc.Loading(html.Div(id="xgb-new-result",
                                 style={"border": "solid", "display": "flex", "align-items": "center",
                                        "flex-direction": "row"}))
        ])
    elif tab == 'tab-svm':
        conf_mat_svm = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_svm), index=True, striped=True,
                                                hover=True, bordered=True, style={"width": "100%"})
        conf_mat_svm.children[0].children[0].children[0].children = ""
        conf_mat_svm.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_svm.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.H3(html.B("Paramètres optimaux")),
            dbc.Table([html.Thead(html.Tr([html.Th("Paramètres", style={"width": "40%"}), html.Th("Valeur")]),
                                  style={"position": "sticky", "top": "0"})] + \
                      [html.Tbody([html.Tr([html.Td("Paramètre de régularisation C"), html.Td("1")]),
                                   html.Tr([html.Td("Kernel utilisé"), html.Td("RBF")]),
                                   html.Tr([html.Td("Paramètre \u03B3"),
                                            html.Td("scale : valeur donnée par  1 / (n_features * X.var()")]),
                                   html.Tr([html.Td("Tolérance pour le critère d'arrêt"), html.Td("0.001")])])],
                      bordered=True,
                      hover=True,
                      dark=True,
                      striped=True, style={"width": "45%"}),
            dcc.Link(html.I("Détails sur les paramètres du modèle SVC"),
                     href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html", target="_blank"),
            html.Br(),
            html.H3(html.B("Performances du modèle")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_svm,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_svm)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(svm_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_svm))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div([html.H3(html.B("Courbe ROC - SVM"), style={"text-align": "center"}), html.Br(),
                                dcc.Graph(
                                    figure=px.area(x=fpr_svm, y=tpr_svm, title=f'ROC Curve (AUC={svm_roc_auc:.3f})',
                                                   labels=dict(x='False Positive Rate',
                                                               y='True Positive Rate')).add_shape(
                                        type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                        margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                    style={"margin-bottom": "10px", "width": "80%"})
                                ],
                               style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Test de nouveaux paramètres")),
            html.Div([
                html.Div([html.P("Fonction kernel"),
                          dcc.Dropdown(["linear", "poly", "rbf", "sigmoid", "precomputed"], value="rbf",
                                       clearable=False, persistence=True, persistence_type="session",
                                       id={"type": "params-svm", "index": "1"})],
                         style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Paramètre C"), dbc.Input(placeholder="Renseigner une valeur...", type="number",
                                                           id={"type": "params-svm", "index": "2"})],
                         style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Paramètre \u03B3"), dbc.Input(placeholder="Renseigner une valeur...", type="text",
                                                                id={"type": "params-svm", "index": "3"})],
                         style={"margin": "0 20px 0 20px"}),
                dcc.Loading(dbc.Button("Lancez le calcul", id="btn-calc-svm", style={"height": "80%"}))
            ], style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Nouveaux résultats obtenus pour le SVM")),
            html.Br(),
            dcc.Loading(html.Div(id="svm-new-result",
                                 style={"border": "solid", "display": "flex", "align-items": "center",
                                        "flex-direction": "row"}))
        ])
    elif tab == 'tab-nn':
        conf_mat_nn = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_nn), index=True, striped=True, hover=True,
                                               bordered=True, style={"width": "100%", "border-radius": "5%"})
        conf_mat_nn.children[0].children[0].children[0].children = ""
        conf_mat_nn.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_nn.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        conf_mat_kaggle = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_kaggle), index=True, striped=True,
                                                   hover=True, bordered=True,
                                                   style={"width": "100%", "border-radius": "5%"})
        conf_mat_kaggle.children[0].children[0].children[0].children = ""
        conf_mat_kaggle.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_kaggle.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        return html.Div([
            html.Br(),
            html.Div([
                html.Div([html.H3(html.B("Paramètres optimaux"), style={"text-align": "center"}),
                          html.Div([dbc.Table([html.Thead(
                              html.Tr([html.Th("Paramètres", style={"text-align": "center", "width": "65%"}),
                                       html.Th("Valeur", style={"text-align": "center"})], style={"width": "25%"}),
                              style={"position": "sticky", "top": "0"})] + \
                                              [html.Tbody([html.Tr(
                                                  [html.Td("Fonction d'activation pour les couches cachées"),
                                                   html.Td("tangeante hyperbolique (tanh)")]),
                                                           html.Tr([html.Td(
                                                               "Solveur utilisé pour l'optimisation des poids"),
                                                                    html.Td("lbfgs (optimiseur quani-newtoniens)")]),
                                                           html.Tr(
                                                               [html.Td("Nombre de couches cachées"), html.Td("2")]),
                                                           html.Tr([html.Td("Nombre de neurones dans la couche 1"),
                                                                    html.Td("4")]),
                                                           html.Tr([html.Td("Nombre de neurones dans la couche 2"),
                                                                    html.Td("3")])])],
                                              bordered=True,
                                              hover=True,
                                              dark=True,
                                              striped=True, style={"overflow": "auto", "width": "100%"}),
                                    dcc.Link(html.I("Détails sur les paramètres du modèle SVC"),
                                             href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
                                             target="_blank")],
                                   style={"overflow": "auto", "height": "250px"}, className="table-wrapper")],
                         style={"width": "50%"}),
                html.Div([html.H3(html.B("Architecture du modèle optimal"), style={"text-align": "center"}),
                          html.Div(html.Img(src="assets/neural_network_architecture.png", id="myImg",
                                            alt="Neural Network Architecture",
                                            style={"width": "100%"}),
                                   style={"border": "solid", "overflow": "auto", "height": "250px"})],
                         style={"display": "flex", "flex-direction": "column", "width": "50%"})
            ], style={"display": "flex", "flex-direction": "row"}),
            dji.Import(src="assets/test-image.js"),
            dcc.Link(html.I("Détails sur les paramètres du modèle MLPClassifier"),
                     href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
                     target="_blank"),
            html.Br(),
            html.H3(html.B("Performances du modèle sur la table credit.xls")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_nn,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_nn)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(nn_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_nn))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div([html.H3(html.B("Courbe ROC - NN"), style={"text-align": "center"}), html.Br(),
                                dcc.Graph(figure=px.area(x=fpr_nn, y=tpr_nn, title=f'ROC Curve (AUC={nn_roc_auc:.3f})',
                                                         labels=dict(x='False Positive Rate',
                                                                     y='True Positive Rate')).add_shape(type='line',
                                                                                                        line=dict(
                                                                                                            dash='dash'),
                                                                                                        x0=0, x1=1,
                                                                                                        y0=0,
                                                                                                        y1=1).update_layout(
                                    margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                          style={"margin-bottom": "10px", "width": "80%"})
                                ],
                               style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.Br(),
            html.Br(),
            html.H3(html.B("Performances du modèle sur la table 'Kaggle: Give me some credit'")),
            html.Div([html.Div([html.H3(html.B("Matrice de confusion")),
                                html.Div(conf_mat_kaggle,
                                         style={"border": "solid", "width": "50%", "display": "inline-block"}),
                                html.Br(),
                                html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_kaggle)), html.Br(),
                                html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(kaggle_roc_auc)),
                                html.Br(),
                                html.H3(html.B("Temps écoulé")),
                                html.H4("{} secondes".format("{} secondes".format(elap_time_kaggle)))],
                               style={"width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"}),
                      html.Div([html.H3(html.B("Courbe ROC - NN - Kaggle"), style={"text-align": "center"}), html.Br(),
                                dcc.Graph(figure=px.area(x=fpr_kaggle, y=tpr_kaggle,
                                                         title=f'ROC Curve (AUC={kaggle_roc_auc:.3f})',
                                                         labels=dict(x='False Positive Rate',
                                                                     y='True Positive Rate')).add_shape(
                                    type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                    margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                          style={"margin-bottom": "10px", "width": "80%"})
                                ],
                               style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                      "align-items": "center"})],
                     style={"border": "solid", "display": "flex", "align-items": "center", "flex-direction": "row"}),
            html.H3(html.B("Test de nouveaux paramètres")),
            html.Div([
                html.Div([html.P("Fonction d'activation"),
                          dcc.Dropdown(["identity", "logistic", "tanh", "relu"], value="tanh", clearable=False,
                                       persistence=True, persistence_type="session",
                                       id="fonc-activ")],
                         style={"margin": "0 20px 0 20px"}),
                html.Div([html.P("Solveur", style={"text-align": "center"}),
                          dcc.Dropdown(["lbfgs", "sgd", "adam"], value="lbfgs", clearable=False, persistence=True,
                                       persistence_type="session", id="solveurs")],
                         style={"margin": "0 20px 0 20px", "width": "8vh"}),
                html.Div([html.P("Nombre de couches cachés"),
                          dcc.Dropdown([i for i in range(1, 11)], clearable=False, value=1, id="nb-hidden-layer",
                                       persistence=True, persistence_type="session")],
                         style={"margin": "0 20px 0 20px"}),
                html.Div(style={"display": "flex", "flex-direction": "row"}, id="nb-neur"),
                dcc.Loading(
                    dbc.Button("Lancer le calcul", id="btn_calc_neur", style={"align-self": "center", "height": "100%"},
                               disabled=False)),
            ], style={"display": "flex", "flex-direction": "row"}),
            html.Br(),
            html.H3(html.B("Architecture du nouveau modèle")),
            html.Div(id="neur-net-div", style={"border": "solid"}),
            html.Div(),
            html.Br(),
            html.H3(html.B("Résultats obtenus pour le nouveau réseau de neurones")),
            dcc.Loading(id="ls-loading-1", children=[html.Div(id="nn-new-result",
                                                              style={"border": "solid", "display": "flex",
                                                                     "align-items": "center",
                                                                     "flex-direction": "row"})],
                        type="default"),
        ])

    # CALCUL AVEC NOUVEAUX PARAMETRES POUR LA REGRESSION LOGISTIQUE PENALISEE


@app.callback(
    Output({"type": "params-log2", "index": "2"}, "options"),
    Input({"type": "params-log2", "index": "1"}, "value"),
)
def update_dbdwn_log2(solv):
    if solv == "newton-cg":
        opts = [{'label': 'none', 'value': 'none', 'disabled': False},
                {'label': 'l1', 'value': 'l1', 'disabled': True},
                {'label': 'l2', 'value': 'l2', 'disabled': False},
                {'label': 'elasticnet', 'value': 'elasticnet', 'disabled': True}]
    elif solv == "lbfgs":
        opts = [{'label': 'none', 'value': 'none', 'disabled': False},
                {'label': 'l1', 'value': 'l1', 'disabled': True},
                {'label': 'l2', 'value': 'l2', 'disabled': False},
                {'label': 'elasticnet', 'value': 'elasticnet', 'disabled': True}]
    elif solv == "liblinear":
        opts = [{'label': 'none', 'value': 'none', 'disabled': True},
                {'label': 'l1', 'value': 'l1', 'disabled': False},
                {'label': 'l2', 'value': 'l2', 'disabled': False},
                {'label': 'elasticnet', 'value': 'elasticnet', 'disabled': True}]
    elif solv == "sag":
        opts = [{'label': 'none', 'value': 'none', 'disabled': False},
                {'label': 'l1', 'value': 'l1', 'disabled': True},
                {'label': 'l2', 'value': 'l2', 'disabled': False},
                {'label': 'elasticnet', 'value': 'elasticnet', 'disabled': True}]
    else:
        opts = [{'label': 'none', 'value': 'none', 'disabled': False},
                {'label': 'l1', 'value': 'l1', 'disabled': False},
                {'label': 'l2', 'value': 'l2', 'disabled': False},
                {'label': 'elasticnet', 'value': 'elasticnet', 'disabled': False}]
    return opts


@app.callback(
    [Output("alert-params-log2", "is_open"),
     Output("alert-calcul-log2", "is_open")],
    [Input("btn-calc-log2", "n_clicks"),
     State({"type": "params-log2", "index": ALL}, "value")]
)
def check_calcul_log2(clic, params_log2):
    if clic is not None:
        if (params_log2[2] is None) or (params_log2[3] is None):
            warn_vis = True
            calc_vis = False
        else:
            warn_vis = False
            calc_vis = True
    else:
        warn_vis = False
        calc_vis = False
    return warn_vis, calc_vis


@app.callback(
    [Output("log2-new-result", "children"),
     Output("new-log2-stored", "data"),
     Output("btn-calc-log2", "disabled")],
    [Input("btn-calc-log2", "n_clicks"),
     State({"type": "params-log2", "index": ALL}, "value"),
     State("new-log2-stored", "data")]
)
def new_calcul_log2(clic, params_log2, log2_res_stored):
    if clic is not None:
        if (params_log2[2] is None) or (params_log2[3] is None):
            ret1 = log2_res_stored
            ret2 = ret1
        else:
            logreg2_new = LogisticRegression(solver=params_log2[0], penalty=params_log2[1], C=params_log2[2],
                                             tol=params_log2[3], max_iter=1000)
            before = datetime.datetime.now()
            logreg2_new.fit(x_train, y_train)
            after = datetime.datetime.now()
            elap_time_log2_new = round((after - before).total_seconds())
            y_pred_log2_new = logreg2_new.predict(x_test)
            accu_rat_log2_new = accuracy_score(y_test, y_pred_log2_new)

            fpr_log2_new, tpr_log2_new, thresholds_log2_new = roc_curve(y_test, logreg2_new.predict_proba(x_test)[:, 1])
            log2_roc_auc_new = roc_auc_score(y_test, logreg2_new.predict_proba(x_test)[:, 1])

            confusion_matrix_log2_new = confusion_matrix(y_test, y_pred_log2_new)
            conf_mat_log2_new = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_log2_new), index=True,
                                                         striped=True, hover=True, bordered=True,
                                                         style={"width": "100%"})
            conf_mat_log2_new.children[0].children[0].children[0].children = ""
            conf_mat_log2_new.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
            conf_mat_log2_new.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
            ret1 = [html.Div([html.H3(html.B("Matrice de confusion")),
                              html.Div(conf_mat_log2_new,
                                       style={"border": "solid", "width": "50%", "display": "inline-block"}), html.Br(),
                              html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_log2_new)), html.Br(),
                              html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(log2_roc_auc_new)),
                              html.Br(),
                              html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_log2_new))],
                             style={"width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"}),
                    html.Div([html.H3(html.B("Courbe ROC - New Penalized Reg Log", style={"text-align": "center"})),
                              html.Br(),
                              dcc.Graph(figure=px.area(x=fpr_log2_new, y=tpr_log2_new,
                                                       title=f'ROC Curve (AUC={log2_roc_auc_new:.3f})',
                                                       labels=dict(x='False Positive Rate',
                                                                   y='True Positive Rate')).add_shape( \
                                  type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                  margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                        style={"margin-bottom": "10px", "width": "80%"})
                              ],
                             style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"})]
            ret2 = ret1
    else:
        ret1 = log2_res_stored
        ret2 = ret1
    return ret1, ret2, False


# CALCUL AVEC NOUVEAUX PARAMETRES POUR L'ARBRE DE DECISION
@app.callback(
    [Output("alert-params-tree", "is_open"),
     Output("alert-calcul-tree", "is_open")],
    [Input("btn-calc-tree", "n_clicks"),
     State({"type": "params-tree", "index": ALL}, "value")]
)
def check_calcul_tree(clic, params_tree):
    if clic is not None:
        if (params_tree[0] is None) or (params_tree[1] is None):
            warn_vis = True
            calc_vis = False
        else:
            warn_vis = False
            calc_vis = True
    else:
        warn_vis = False
        calc_vis = False
    return warn_vis, calc_vis


@app.callback(
    [Output("tree-new-result", "children"),
     Output("new-tree-stored", "data"),
     Output("btn-calc-tree", "disabled")],
    [Input("btn-calc-tree", "n_clicks"),
     State({"type": "params-tree", "index": ALL}, "value"),
     State("new-tree-stored", "data")]
)
def new_calcul_tree(clic, params_tree, tree_res_stored):
    if clic is not None:
        if (params_tree[0] is None) or (params_tree[1] is None):
            ret1 = tree_res_stored
            ret2 = ret1
        else:
            dt_new = DecisionTreeClassifier(criterion='gini', max_depth=params_tree[0], min_impurity_decrease=0.0001,
                                            min_samples_leaf=params_tree[1], random_state=3)
            before = datetime.datetime.now()
            dt_new.fit(x_train, y_train)
            after = datetime.datetime.now()
            elap_time_dt_new = round((after - before).total_seconds())
            y_pred_dt_new = dt_new.predict(x_test)
            accu_rat_dt_new = accuracy_score(y_test, y_pred_dt_new)

            fpr_dt_new, tpr_dt_new, thresholds_dt_new = roc_curve(y_test, dt_new.predict_proba(x_test)[:, 1])
            dt_roc_auc_new = roc_auc_score(y_test, dt_new.predict_proba(x_test)[:, 1])

            confusion_matrix_dt_new = confusion_matrix(y_test, y_pred_dt_new)
            conf_mat_dt_new = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_dt_new), index=True, striped=True,
                                                       hover=True, bordered=True, style={"width": "100%"})
            conf_mat_dt_new.children[0].children[0].children[0].children = ""
            conf_mat_dt_new.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
            conf_mat_dt_new.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
            ret1 = [html.Div([html.H3(html.B("Matrice de confusion")),
                              html.Div(conf_mat_dt_new,
                                       style={"border": "solid", "width": "50%", "display": "inline-block"}), html.Br(),
                              html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_dt_new)), html.Br(),
                              html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(dt_roc_auc_new)),
                              html.Br(),
                              html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_dt_new))],
                             style={"width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"}),
                    html.Div([html.H3(html.B("Courbe ROC - New Tree", style={"text-align": "center"})), html.Br(),
                              dcc.Graph(figure=px.area(x=fpr_dt_new, y=tpr_dt_new,
                                                       title=f'ROC Curve (AUC={dt_roc_auc_new:.3f})',
                                                       labels=dict(x='False Positive Rate',
                                                                   y='True Positive Rate')).add_shape( \
                                  type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                  margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                        style={"margin-bottom": "10px", "width": "80%"})
                              ],
                             style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"})]
            ret2 = ret1
    else:
        ret1 = tree_res_stored
        ret2 = ret1
    return ret1, ret2, False


# CALCUL AVEC NOUVEAUX PARAMETRES POUR LE RANDOM FOREST
@app.callback(
    Output("alert-calcul-rf", "is_open"),
    Input("btn-calc-rf", "n_clicks")
)
def check_calcul_rf(clic):
    if clic is not None:
        calc_vis = True
    else:
        calc_vis = False
    return calc_vis


@app.callback(
    [Output("rf-new-result", "children"),
     Output("new-rf-stored", "data"),
     Output("btn-calc-rf", "disabled")],
    [Input("btn-calc-rf", "n_clicks"),
     State({"type": "params-rf", "index": ALL}, "value"),
     State("new-rf-stored", "data")]
)
def new_calcul_tree(clic, params_rf, rf_res_stored):
    if clic is not None:
        if (params_rf[0] is None) or (params_rf[1] is None):
            ret1 = rf_res_stored
            ret2 = ret1
        else:
            rf_new = RandomForestClassifier(n_estimators=params_rf[0], criterion='gini', min_impurity_decrease=0.0001,
                                            random_state=13, max_depth=params_rf[1])

            before = datetime.datetime.now()
            rf_new.fit(x_train, y_train)
            after = datetime.datetime.now()
            elap_time_rf_new = round((after - before).total_seconds())
            y_pred_rf_new = rf_new.predict(x_test)
            accu_rat_rf_new = accuracy_score(y_test, y_pred_rf_new)

            fpr_rf_new, tpr_rf_new, thresholds_rf_new = roc_curve(y_test, rf_new.predict_proba(x_test)[:, 1])
            rf_roc_auc_new = roc_auc_score(y_test, rf_new.predict_proba(x_test)[:, 1])

            confusion_matrix_rf_new = confusion_matrix(y_test, y_pred_rf_new)
            conf_mat_rf_new = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_rf_new), index=True, striped=True,
                                                       hover=True, bordered=True, style={"width": "100%"})
            conf_mat_rf_new.children[0].children[0].children[0].children = ""
            conf_mat_rf_new.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
            conf_mat_rf_new.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
            ret1 = [html.Div([html.H3(html.B("Matrice de confusion")),
                              html.Div(conf_mat_rf_new,
                                       style={"border": "solid", "width": "50%", "display": "inline-block"}), html.Br(),
                              html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_rf_new)), html.Br(),
                              html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(rf_roc_auc_new)),
                              html.Br(),
                              html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_rf_new))],
                             style={"width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"}),
                    html.Div([html.H3(html.B("Courbe ROC - New RF", style={"text-align": "center"})), html.Br(),
                              dcc.Graph(figure=px.area(x=fpr_rf_new, y=tpr_rf_new,
                                                       title=f'ROC Curve (AUC={rf_roc_auc_new:.3f})',
                                                       labels=dict(x='False Positive Rate',
                                                                   y='True Positive Rate')).add_shape( \
                                  type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                  margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                        style={"margin-bottom": "10px", "width": "80%"})
                              ],
                             style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"})]
            ret2 = ret1
    else:
        ret1 = rf_res_stored
        ret2 = ret1
    return ret1, ret2, False


# CALCUL AVEC NOUVEAUX PARAMETRES POUR LE XGB
@app.callback(
    [Output("alert-params-xgb", "is_open"),
     Output("alert-calcul-xgb", "is_open")],
    [Input("btn-calc-xgb", "n_clicks"),
     State({"type": "params-xgb", "index": ALL}, "value")]
)
def check_calcul_xgb(clic, params_xgb):
    if clic is not None:
        if (params_xgb[2] is None) or (params_xgb[3] is None):
            warn_vis = True
            calc_vis = False
        else:
            warn_vis = False
            calc_vis = True
    else:
        warn_vis = False
        calc_vis = False
    return warn_vis, calc_vis


@app.callback(
    [Output("xgb-new-result", "children"),
     Output("new-xgb-stored", "data"),
     Output("btn-calc-xgb", "disabled")],
    [Input("btn-calc-xgb", "n_clicks"),
     State({"type": "params-xgb", "index": ALL}, "value"),
     State("new-xgb-stored", "data")]
)
def new_calcul_tree(clic, params_xgb, xgb_res_stored):
    if clic is not None:
        if (params_xgb[2] is None) or (params_xgb[3] is None):
            ret1 = xgb_res_stored
            ret2 = ret1
        else:

            xgb_new = xgb.XGBClassifier(objective='binary:logistic', n_estimators=params_xgb[0],
                                        max_depth=params_xgb[1], colsample_bytree=params_xgb[2],
                                        learning_rate=params_xgb[3], alpha=19, random_state=13)

            before = datetime.datetime.now()
            xgb_new.fit(x_train, y_train)
            after = datetime.datetime.now()

            elap_time_xgb_new = round((after - before).total_seconds())
            y_pred_xgb_new = xgb_new.predict(x_test)
            accu_rat_xgb_new = accuracy_score(y_test, y_pred_xgb_new)

            fpr_xgb_new, tpr_xgb_new, thresholds_xgb_new = roc_curve(y_test, xgb_new.predict_proba(x_test)[:, 1])
            xgb_roc_auc_new = roc_auc_score(y_test, xgb_new.predict_proba(x_test)[:, 1])

            confusion_matrix_xgb_new = confusion_matrix(y_test, y_pred_xgb_new)
            conf_mat_xgb_new = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_xgb_new), index=True,
                                                        striped=True, hover=True, bordered=True,
                                                        style={"width": "100%"})
            conf_mat_xgb_new.children[0].children[0].children[0].children = ""
            conf_mat_xgb_new.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
            conf_mat_xgb_new.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
            ret1 = [html.Div([html.H3(html.B("Matrice de confusion")),
                              html.Div(conf_mat_xgb_new,
                                       style={"border": "solid", "width": "50%", "display": "inline-block"}), html.Br(),
                              html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_xgb_new)), html.Br(),
                              html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(xgb_roc_auc_new)),
                              html.Br(),
                              html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_xgb_new))],
                             style={"width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"}),
                    html.Div([html.H3(html.B("Courbe ROC - New XGB", style={"text-align": "center"})), html.Br(),
                              dcc.Graph(figure=px.area(x=fpr_xgb_new, y=tpr_xgb_new,
                                                       title=f'ROC Curve (AUC={xgb_roc_auc_new:.3f})',
                                                       labels=dict(x='False Positive Rate',
                                                                   y='True Positive Rate')).add_shape( \
                                  type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                  margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                        style={"margin-bottom": "10px", "width": "80%"})
                              ],
                             style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"})]
            ret2 = ret1
    else:
        ret1 = xgb_res_stored
        ret2 = ret1
    return ret1, ret2, False


# CALCUL AVEC NOUVEAUX PARAMETRES POUR LE SVM
@app.callback(
    [Output("alert-params-svm", "is_open"),
     Output("alert-calcul-svm", "is_open")],
    [Input("btn-calc-svm", "n_clicks"),
     State({"type": "params-svm", "index": ALL}, "value")]
)
def check_calcul_svm(clic, params_svm):
    if clic is not None:
        if (params_svm[1] is None) or (params_svm[2] is None):
            warn_vis = True
            calc_vis = False
        else:
            warn_vis = False
            calc_vis = True
    else:
        warn_vis = False
        calc_vis = False
    return warn_vis, calc_vis


@app.callback(
    [Output("svm-new-result", "children"),
     Output("new-svm-stored", "data"),
     Output("btn-calc-svm", "disabled")],
    [Input("btn-calc-svm", "n_clicks"),
     State({"type": "params-svm", "index": ALL}, "value"),
     State("new-svm-stored", "data")]
)
def new_calcul_tree(clic, params_svm, svm_res_stored):
    if clic is not None:
        if (params_svm[1] is None) or (params_svm[2] is None):
            ret1 = svm_res_stored
            ret2 = ret1
        else:
            svm_new = svm.SVC(kernel=params_svm[0], C=params_svm[1], gamma=params_svm[2], probability=True)
            before = datetime.datetime.now()
            svm_new.fit(x_train, y_train)
            after = datetime.datetime.now()
            elap_time_svm_new = round((after - before).total_seconds())
            y_pred_svm_new = svm_new.predict(x_test)
            accu_rat_svm_new = accuracy_score(y_test, y_pred_svm_new)

            fpr_svm_new, tpr_svm_new, thresholds_svm_new = roc_curve(y_test, svm_new.predict_proba(x_test)[:, 1])
            svm_roc_auc_new = roc_auc_score(y_test, svm_new.predict_proba(x_test)[:, 1])

            confusion_matrix_svm_new = confusion_matrix(y_test, y_pred_svm_new)
            conf_mat_svm_new = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_svm_new), index=True,
                                                        striped=True, hover=True, bordered=True,
                                                        style={"width": "100%"})
            conf_mat_svm_new.children[0].children[0].children[0].children = ""
            conf_mat_svm_new.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
            conf_mat_svm_new.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
            ret1 = [html.Div([html.H3(html.B("Matrice de confusion")),
                              html.Div(conf_mat_svm_new,
                                       style={"border": "solid", "width": "50%", "display": "inline-block"}), html.Br(),
                              html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_svm_new)), html.Br(),
                              html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(svm_roc_auc_new)),
                              html.Br(),
                              html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_svm_new))],
                             style={"width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"}),
                    html.Div([html.H3(html.B("Courbe ROC - New SVM", style={"text-align": "center"})), html.Br(),
                              dcc.Graph(figure=px.area(x=fpr_svm_new, y=tpr_svm_new,
                                                       title=f'ROC Curve (AUC={svm_roc_auc_new:.3f})',
                                                       labels=dict(x='False Positive Rate',
                                                                   y='True Positive Rate')).add_shape( \
                                  type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                  margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                                        style={"margin-bottom": "10px", "width": "80%"})
                              ],
                             style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                    "align-items": "center"})]
            ret2 = ret1
    else:
        ret1 = svm_res_stored
        ret2 = ret1
    return ret1, ret2, False


# CALCUL AVEC NOUVEAUX PARAMETRES POUR LE RESEAU DE NEURONES

@app.callback(
    Output("alert-calcul-nn", "is_open"),
    Input("btn_calc_neur", "n_clicks")
)
def check_calcul_nn(clic):
    if clic is not None:
        calc_vis = True
    else:
        calc_vis = False
    return calc_vis

@app.callback(
    Output("nb-neur", "children"),
    Input("nb-hidden-layer", "value"),
    State("nb-neur", "children"),
    State("nb-hidden-layer", "n_clicks")
)
def update_output_div_nn(nb, current_children, nclic):
    if nb != 0:
        new_children = [html.Div([html.P(f"Couche n°{i}"), dcc.Dropdown(id={"type": "layers", "index": "L_" + str(i)},
                                                                        options=[j for j in range(1, 101)], value=1,
                                                                        clearable=False,
                                                                        persistence=True, persistence_type="session")],
                                 style={"margin": "0 20px 0 20px", "text-align": "center"}) for i in
                        range(1, int(nb) + 1)]
        current_children = new_children
    else:
        current_children = current_children
    return current_children


@app.callback(
    [Output("nn-new-result", "children"), Output("new-nn-stored", "data"), Output("btn_calc_neur", "disabled")],
    [Input("btn_calc_neur", "n_clicks"), State("nb-neur", "children"), State({"type": "layers", "index": ALL}, "value"),
     State("new-nn-stored", "data"), State("fonc-activ", "value"), State("solveurs", "value")]
    #    [Input("btn_calc_neur", "n_clicks"),Input("nb-neur", "children"),State("nn-new-result", "children"),State("new-nn-stored","data")],
)
def update_new_nn_results(clic, val, vals, nn_res_stored, func_act, solv):
    if clic is not None:
        layers = vals
        clf_model_nn_new = neural_network.MLPClassifier(hidden_layer_sizes=np.array(layers), activation=func_act,
                                                        solver=solv, max_iter=10000, alpha=1e-5, tol=1e-5)
        before = datetime.datetime.now()
        clf_model_nn_new.fit(x_train, y_train)
        after = datetime.datetime.now()
        elap_time_nn_new = round((after - before).total_seconds())
        y_pred_nn_new = clf_model_nn_new.predict(x_test)
        accu_rat_nn_new = accuracy_score(y_test, y_pred_nn_new)
        fpr_nn_new, tpr_nn_new, thresholds_nn_new = roc_curve(y_test, clf_model_nn_new.predict_proba(x_test)[:, 1])
        nn_roc_auc_new = roc_auc_score(y_test, clf_model_nn_new.predict_proba(x_test)[:, 1])
        confusion_matrix_nn_new = confusion_matrix(y_test, y_pred_nn_new)
        conf_mat_nn_new = dbc.Table.from_dataframe(pd.DataFrame(confusion_matrix_nn_new), index=True, striped=True,
                                                   hover=True, bordered=True, style={"width": "100%"})
        conf_mat_nn_new.children[0].children[0].children[0].children = ""
        conf_mat_nn_new.children[1].children[0].children[0] = html.Td("0", style={"font-weight": "bold"})
        conf_mat_nn_new.children[1].children[1].children[0] = html.Td("1", style={"font-weight": "bold"})
        ret1 = [html.Div([html.H3(html.B("Matrice de confusion")),
                          html.Div(conf_mat_nn_new,
                                   style={"border": "solid", "width": "50%", "display": "inline-block"}), html.Br(),
                          html.H3(html.B("Accuracy ratio")), html.H4("{:.3f}".format(accu_rat_nn_new)), html.Br(),
                          html.H3(html.B("Area Under the Curve")), html.H4("{:.3f}".format(nn_roc_auc_new)), html.Br(),
                          html.H3(html.B("Temps écoulé")), html.H4("{} secondes".format(elap_time_nn_new))],
                         style={"width": "50%", "display": "flex", "flex-direction": "column",
                                "align-items": "center"}),
                html.Div([html.H3(html.B("Courbe ROC - New Neural Network"), style={"text-align": "center"}), html.Br(),
                          dcc.Graph(
                              figure=px.area(x=fpr_nn_new, y=tpr_nn_new, title=f'ROC Curve (AUC={nn_roc_auc_new:.3f})',
                                             labels=dict(x='False Positive Rate', y='True Positive Rate')).add_shape( \
                                  type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1).update_layout(
                                  margin=dict(l=20, r=2, t=4, b=20), title_y=0.95, title_x=0.5),
                              style={"margin-bottom": "10px", "width": "80%"})
                          ],
                         style={"height": "100%", "width": "50%", "display": "flex", "flex-direction": "column",
                                "align-items": "center"})]
        ret2 = ret1
    else:
        ret1 = nn_res_stored
        ret2 = ret1
    return ret1, ret2, False


@app.callback(
    Output("neur-net-div", "children"),
    [Input({"type": "layers", "index": ALL}, "value"), State({"type": "layers", "index": ALL}, "value"),
     Input("nb-neur", "children")]
)
def update_new_layers(val, valinput, chil):
    if len(chil) > -1:
        nb_layers = len(val)
        pos_out = 100 * (nb_layers + 1)
        inputs = [
            {
                'data': {'id': "inputs", 'label': "74 inputs"},
                'locked': True, "classes": "node_par"
            },
            {
                'data': {'id': "a_11", 'label': "x1", "parent": "inputs"},
                'locked': True, 'position': {'x': 0, 'y': 100}, "classes": "node_chil green"
            },
            {
                'data': {'id': "a_21", 'label': "...", "parent": "inputs"},
                'locked': True, 'position': {'x': 0, 'y': 200}, "classes": "node_chil green"
            },
            {
                'data': {'id': "a_31", 'label': "x\N{SUBSCRIPT SEVEN}\N{SUBSCRIPT FOUR}", "parent": "inputs"},
                'locked': True, 'position': {'x': 0, 'y': 300}, "classes": "node_chil green"
            }
        ]
        output = [
            {'data': {'id': "output", 'label': "default payment"}, 'locked': True, "classes": "node_par"},
            {'data': {'id': "a_1" + str(nb_layers + 2), 'label': "y", "parent": "output"}, 'locked': True,
             'position': {'x': pos_out, 'y': 200}, "classes": "node_chil red"}
        ]

        # Parent Nodes
        parents = [{'data': {'id': 'L_{}'.format(lay), 'label': 'Couche n°{}'.format(lay)}, "locked": True,
                    "classes": "node_par"} for lay in range(1, nb_layers + 1)]

        nodes = [
            {'data': {'id': 'a_{}{}'.format(j, lay + 1), 'label': 'a_{}{}'.format(j, lay + 1),
                      "parent": "L_{}".format(lay)}, "locked": True,
             'position': {'x': int('{}'.format(100 * lay)), 'y': int('{}'.format(100 * j))},
             "classes": "node_chil blue"} for lay in range(1, nb_layers + 1) for j in range(1, val[lay - 1] + 1)
        ]
        val_ext = [3] + val + [1]
        edges = [{'data': {'source': 'a_{}{}'.format(node1, lay + 1), 'target': 'a_{}{}'.format(node2, lay + 2)}} for
                 lay in range(nb_layers + 1) for node1 in range(1, val_ext[lay] + 1) for node2 in
                 range(1, val_ext[lay + 1] + 1)]
        elts = inputs + parents + output + nodes + edges
        cytosc = cyto.Cytoscape(id='cytoscape-compound',
                                layout={'name': 'preset', "animate": True},
                                stylesheet=my_stylesheet,
                                style={'width': '100%', 'height': '50vh'},
                                elements=elts,
                                autoRefreshLayout=True,
                                )
    return cytosc


if __name__ == "__main__":
    app.run_server()
