import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,average_precision_score, confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import joblib
import os
from pathlib import Path

#Github stuff
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

demo_path    = DATA_DIR / "DEMO_L.xpt"
alb_path     = DATA_DIR / "ALB_CR_L.xpt"
biochem_path = DATA_DIR / "BIOPRO_L.xpt"




#-------------------------------------------------------------------
# Load and Merge Data
#-------------------------------------------------------------------



#Loading our Files 
demo    = pd.read_sas(str(demo_path))
alb_cre = pd.read_sas(str(alb_path))
biochem = pd.read_sas(str(biochem_path))


#This will merge all of out data sets according to the SEQN id. The "left" means that no data will be lost,
#and SEQN's that do not have data in all three sets will have a blank cell in place of the missing test. 
#This makes the data still usable
df = demo.merge(alb_cre, on="SEQN", how="left").merge(biochem, on="SEQN", how="left")

#Will filter out all SEQN's of minors
df = df[df["RIDAGEYR"] >= 18]

#Done as a quick check to see how it looks
#print(df.head())
#print(df.shape)





#-------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------




#First I'll do UACR (albumin-to-creatinine ratio in mg/g)
#100 × (urine albumin [mg/L] ÷ urine creatinine [mg/dL])
#This will make a new column for each SEQN's UACR value
df["UACR"] = 100 * (df["URXUMA"] / df["URXUCR"])

#Next I'll do eGFR using CKD-EPI
#Formula is too complicated to type here, but can be found in as reference 7
#This is a lot more complicated than the one for UACR,
def calc_egfr(scr, age, sex):
    kappa = 0.7 if sex == 2 else 0.9  # F=2, M=1
    alpha = -0.241 if sex == 2 else -0.302
    min_part = min(scr / kappa, 1) ** alpha
    max_part = max(scr / kappa, 1) ** -1.2
    sex_factor = 1.012 if sex == 2 else 1.0
    return 142 * min_part * max_part * (0.9938 ** age) * sex_factor

#This was really confusing to make. The df["eGFR] makes a new row for the eGFR.
#df.apply with axis=1 means that we are doing this to every row in the datset
#lambda row means that we are manipulating a single row at a time, and we are defining the values that are going into
#the eGFR equation.
df["eGFR"] = df.apply(lambda row: calc_egfr(row["LBXSCR"], row["RIDAGEYR"], row["RIAGENDR"]), axis=1)

#Makes a new row for if they have CKD or not.
#((df["eGFR"] < 60) | (df["UACR"] >= 30)) is an OR function that will assign a TRUE if either condition is correct.
#The .astype(int) turns trues to 1 and anything else to 0
df["CKD"] = ((df["eGFR"] < 60) | (df["UACR"] >= 30)).astype(int)

#quick checks
#print(df[["SEQN", "RIDAGEYR", "RIAGENDR", "LBXSCR", "eGFR", "UACR", "CKD"]].head())  ###This shows the inputs that I care about
#print(df["CKD"].mean() * 100, "%")  ### This is to make sure we get a realistic CKD prevelance. I got 13.29%, which is within normal ranges



#-------------------------------------------------------------------
# Checking Data Quality
#-------------------------------------------------------------------



#This will make sure that prevelence is onlyu considered with people who have BOTH an eGFR and a UACR, .notna will make all unknown cells FALSE
have_both = df["eGFR"].notna() & df["UACR"].notna()

#This is just calculating prevelance 
#.loc is used to locate a column by its string name, and not integer location
ckd_prevalence = df.loc[have_both, "CKD"].mean() * 100

print("\nSample Size")
print(f"Adults total: {len(df):,}") #The comma adds commas to the printed number
print(f"Amount with UACR and eGFR values: {have_both.sum():,}")

print("\nCKD Prevelance with people who have UACR & eGFR")
print(f"{ckd_prevalence:.2f}%")

#Summary statistics for key variables as a percentile
summary_cols = ["RIDAGEYR", "eGFR", "UACR"]
print("\nSummary Stats of Key Variables")
print(df[summary_cols] .rename(columns={"RIDAGEYR": "Age"}).describe(percentiles=[0.05, 0.10, 0.25, 0.5, 0.75, 0.95])) #shows 5th, 10th,...,95th percentile

#Summary based on CKD status
print("\nSummary based on CKD Status")
print(
    df.loc[have_both, ["CKD", "RIDAGEYR", "eGFR", "UACR"]]
      .rename(columns={"RIDAGEYR": "Age"})
      .groupby("CKD") #Splits into a group with CKD, and one without
      .agg({"Age": ["count", "mean", "median"],
          "eGFR": ["mean", "median"],
          "UACR": ["mean", "median"]}))


#This is to natural log transform the UACR data that is usually very skewed
df["logUACR"] = np.log1p(df["UACR"]) 

#This will flag obvious outliers
df["UACR_outlier"] = (df["UACR"] < 0) | (df["UACR"] > 100000)
df["eGFR_outlier"] = (df["eGFR"] < 0) | (df["eGFR"] > 180)

print("\nOutliers")
print("UACR impossible values:", int(df["UACR_outlier"].sum()))
print("eGFR impossible values:", int(df["eGFR_outlier"].sum()))
#I do not expect the NHANES data to have outliers, because the CDC would have most likely removed them prior. I have this for if any new data is used that has not been pre-filtered by the CDC

#eGFR histogram
plt.figure()    #Makes Blank Figure   
df["eGFR"].dropna().plot(kind="hist", bins=40)   #Makes the histogram
plt.title("eGFR Distribution")  #Title
plt.xlabel("eGFR (mL/min)") #x-axis title
plt.ylabel("Count")      #y-axis title
plt.tight_layout()
plt.show()

#UACR histogram pre-log-transformation
plt.figure()
df["UACR"].dropna().plot(kind="hist", bins=60)
plt.title("UACR Distribution (raw)")
plt.xlabel("UACR (mg/g)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#UACR histogram log-transformed
plt.figure()
df["logUACR"].dropna().plot(kind="hist", bins=60)
plt.title("log1p(UACR) Distribution")
plt.xlabel("log1p(UACR)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


#This is defining the important variables that will be included in the exported CSV file
out_cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "DMDEDUC2",
    "URXUMA", "URXUCR", "UACR", "logUACR","LBXSCR", "eGFR", "CKD"]

#This is to change the NHANES variable names to what they represent
rename_NHANES = {"SEQN": "ID",
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",       
    "RIDRETH3": "Race_or_Ethnicity",
    "INDFMPIR": "Income_Poverty_Ratio",
    "DMDEDUC2": "Education_Level",
    "URXUMA": "Urine_Albumin",
    "URXUCR": "Urine_Creatinine",
    "logUACR": "logUACR",
    "LBXSCR": "Serum_Creatinine",}


#Making a name for smaller set
export_data = df[out_cols].rename(columns=rename_NHANES)

#This will change the sex and race from their binary code to their string name
export_data["Sex"] = export_data["Sex"].astype("Int64").map({1: "Male", 2: "Female"})
export_data["Race_or_Ethnicity"] = export_data["Race_or_Ethnicity"].astype("Int64").map({
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-Racial"})   #The data automatically exports the race as a binary number, and this line turns that number back into the string it represents




out_csv_path = RESULTS_DIR / "ckd_analysis_ready.csv"
export_data.to_csv(out_csv_path, index=False)
print(f"\nSaved: {out_csv_path}")





#-------------------------------------------------------------------
# Feature Set & Model Development
#-------------------------------------------------------------------



#Choose features for the model
feature_cols = ["Age", 
    "Sex", 
    "eGFR", 
    "logUACR", 
    "Race_or_Ethnicity", 
    "Income_Poverty_Ratio", 
    "Education_Level"]

#Remove rows where CKD , eGFR, or logUACR are not present
model_data = export_data.dropna(subset=["CKD", "eGFR", "logUACR"])

#This define inputs as X and targets as y to make coding easier
X = model_data[feature_cols]
Y= model_data["CKD"]

#This is the one hot encoding where we will use 0/1's that the model can use
X = pd.get_dummies(X, columns=["Sex", "Race_or_Ethnicity", "Education_Level"])

#This splits the datafram into training data and testing data, which will be 80/20 respectively.
#The stratification means that both sets will have the same proportion of CKD positive people, which is around 13%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

#These are to make sure everything looks correct
print("\nInputs")
print("Shape:", X.shape)
print("Columns:", list(X.columns))
print("\nTrain/Test Split")
print("Training size:", X_train.shape[0])
print("Testing size:", X_test.shape[0])
print("\nCKD prevalence (full dataset):", round(Y.mean()*100, 2), "%")
print("CKD prevalence (train):", round(Y_train.mean()*100, 2), "%")
print("CKD prevalence (test):", round(Y_test.mean()*100, 2), "%")


#This is the first step in actually building the model. Balanced gives weight to the minority CKD, and lbfgs is the algorithm to find the best coefficients for the inputs
#logreg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")

#This is telling the model to use only the training data set, and .fit is telling to to estimate the best coefficients for predicting CKD
#logreg.fit(X_train, Y_train)

#this is asking the models to return probabilities instead of binary, and we are only focusing on when CKD=1, meaning whtn they have it
#In total, this will just give an array of the probability for having CKD for all 4k people in test set
#y_proba = logreg.predict_proba(X_test)[:, 1]


#threshold = 0.35  #This means that anything above a 35% chance will be marked as CKD positive. I made it kind of low because I think the model will work a little better if its more sensitice
#y_pred = (y_proba >= threshold).astype(int)     #This will go through our array of percentages and make them a 0 for no or a 1 for yes

#roc_auc = roc_auc_score(Y_test, y_proba)    #How well model seperates CKD vs non-CKD; 1 is good and ranges to 0.5 which is useless
#pr_auc  = average_precision_score(Y_test, y_proba)  #Measueres how good the model is at finding CKD positive cases, a higher number is better
#cm      = confusion_matrix(Y_test, y_pred)      #Gives 2x2 table that 

#print("\nModel Performance for the Test Set")
#print(f"ROC-AUC: {roc_auc:.3f}")
#print(f"PR-AUC : {pr_auc:.3f}")
#print("\nConfusion Matrix @ threshold =", threshold)
#print(cm)
#print("\nClassification Report @ threshold =", threshold)
#print(classification_report(Y_test, y_pred, digits=3))

#At this point I am running into an error, because the model can not handle NaN values directly. I've been searching this up for a bit and it says I can either
#delete rows with a missing value, which I do not want to do because I will lost a lot of important data, or do somethijng with a SimpleImputer, which is a 
#function I know very little about. Because I am just encroaching over the 4 hour mark, I think that it is best that I put a pin on this until further notice.
#Additional comments in report.