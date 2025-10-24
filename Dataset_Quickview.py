import pandas as pd

demo = pd.read_sas(r"Brown and Morgun-Shulzhenko\DEMO_L.xpt")
alb_cre= pd.read_sas(r"Brown and Morgun-Shulzhenko\ALB_CR_L.xpt")
biochem = pd.read_sas(r"Brown and Morgun-Shulzhenko\BIOPRO_L.xpt")

# .head shows first 5 rows of each dataset
print("DEMO (Demographics):")
print(demo.head(), "\n")

print("ALB_CR (Albumin/Creatinine):")
print(alb_cre.head(), "\n")

print("BIOPRO (Biochemistry):")
print(biochem.head(), "\n")


#The demographics and Biochem Profile have too many columns to show in the terminal straight up.
#Variables we care about are as follows. (Variable nemes can be found linked in references)
#
#   Demographics: 
#   SEQN - participant ID 
#   RIDAGEYR - age 
#   RIAGENDR - sex
#   RIDRETH3 - race/ethnicity 
#   INDFMPIR - income-to-poverty ratio 
#   DMDEDUC2 - education level 
#
#   Alb_Cre:
#   URXUMA - urine albumin (mg/L)
#   URXUCR - urine creatinine (mg/dL)
#
#   Biochem:
#   LBXSCR - serum creatinine