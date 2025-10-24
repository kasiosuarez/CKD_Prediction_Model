CKD_Prediction_Model.py   # Main Python script
requirements.txt          # List of dependencies

Setup Instructions
1.Clone the repository by pasting this in terminal

2. Install dependencies
   pip install -r requirements.txt

3. Prepare Data Folder
Create a folder called data in the project root.
Download the following NHANES .xpt datasets from the CDC website and place them inside data:
DEMO_L.xpt

ALB_CR_L.xpt (Urine Albumin & Creatinine)

BIOPRO_L.xpt (Biochemistry Profile, includes Serum Creatinine)

(4,5, and 12 in references)

4. Run the script
python CKD_Prediction_Model.py
