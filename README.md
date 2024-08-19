# dacon_drug_design

## original data and Preprocessed data

- [Google drive link](https://drive.google.com/drive/folders/1SUATfOOw7MedfSlIak3yzMNTPzIy0SvY?usp=sharing)<br/>

저장 형식

- `train_data_img.csv` - train_data에 image_feature_vector column에 추가한 csv  
- `val_data_img.csv` - valdiation data에 image_feature_vector column에 추가한 csv

- `train_data_smi_img.csv` - train_data에 image, smi feature vector column에 추가한 csv  
- `val_data_smi_img.csv` - valdiation data에 image, smi feature vector column에 추가한 csv

- `train_data_smi_img_protein.csv` - train_data에 image, smi, protein feature vector column에 추가한 csv  
- `val_data_smi_img_protein.csv` - valdiation data에 image, smi, protein feature vector column에 추가한 csv

## 1. Bert settings for smiles feature vector
[chemberta repository](https://github.com/seyonechithrananda/bert-loves-chemistry)
environment setting and usage follow the chemberta repository

## 2. OCSR settings for molecular image feature vector
[MolScribe repository](https://github.com/thomas0809/MolScribe)
environment setting and usage follow the MolScribe repository

## 3. Molecular Properties settings for molecular properties feature vector
Molecular Properties settings are the same as the model training environment and are implemented through the rdkit library.
No separate preprocessing is required
The environment file is linked here. [environment](https://github.com/jjjabcd/dacon_drug_design/blob/main/ocsr_model/environment.yml)

## 4. target Protein embedding settings for target protein embedding vector
