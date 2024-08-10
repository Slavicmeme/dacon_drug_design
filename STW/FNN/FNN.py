import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Fragments
import numpy as np
import pandas as pd

# 장치 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MolecularDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.smiles_data = self.df['Smiles'].dropna()
        self.features = np.array([self.calculate_properties(smiles) for smiles in self.smiles_data])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    @staticmethod
    def calculate_properties(smiles):
        mol = Chem.MolFromSmiles(smiles)
        
        properties = []
        
        properties.append(Descriptors.MolWt(mol))  # 분자량 (Molecular Weight)
        properties.append(Crippen.MolLogP(mol))  # Crippen의 방식으로 계산된 로그 P 값 (LogP)
        properties.append(Descriptors.TPSA(mol))  # 극성 표면적 (Topological Polar Surface Area)
        properties.append(Lipinski.NumHAcceptors(mol))  # 수소 수용체의 개수 (Number of Hydrogen Bond Acceptors)
        properties.append(Lipinski.NumHDonors(mol))  # 수소 공여체의 개수 (Number of Hydrogen Bond Donors)
        properties.append(Lipinski.NumRotatableBonds(mol))  # 회전 가능한 결합의 수 (Number of Rotatable Bonds)
        properties.append(Chem.GetFormalCharge(mol))  # 분자의 형식적 전하 (Formal Charge)
        properties.append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))  # 원자 중심 입체 중심 수 (Number of Atom Stereocenters)
        properties.append(rdMolDescriptors.CalcFractionCSP3(mol))  # 탄소 sp3 부분의 분율 (Fraction of sp3 Carbon Atoms)
        properties.append(Descriptors.NumAliphaticCarbocycles(mol))  # 지방족 탄소고리의 수 (Number of Aliphatic Carbocycles)
        properties.append(Descriptors.NumAromaticRings(mol))  # 방향족 고리의 수 (Number of Aromatic Rings)
        properties.append(Descriptors.NumHeteroatoms(mol))  # 헤테로 원자의 수 (Number of Heteroatoms)
        properties.append(Fragments.fr_COO(mol))  # 카복실산 작용기의 수 (Number of Carboxylic Acid Groups)
        properties.append(Fragments.fr_Al_OH(mol))  # 알코올 작용기의 수 (Number of Aliphatic Alcohol Groups)
        properties.append(Fragments.fr_alkyl_halide(mol))  # 알킬 할라이드 작용기의 수 (Number of Alkyl Halide Groups)
        properties.append(Descriptors.NumAromaticCarbocycles(mol))  # 방향족 탄소고리의 수 (Number of Aromatic Carbocycles)
        properties.append(Fragments.fr_piperdine(mol))  # 피페리딘 작용기의 수 (Number of Piperidine Groups)
        properties.append(Fragments.fr_methoxy(mol))  # 메톡시 작용기의 수 (Number of Methoxy Groups)

        return properties

class AdvancedFNNModel(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedFNNModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(32, input_dim)  # 또는 output_dim으로 설정 가능
        self.bn4 = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.bn4(self.fc4(x))  # 마지막 층에는 활성화 함수가 없을 수도 있습니다
        return x

def extract_features_from_csv(csv_file):
    # 데이터셋 로드 및 모델 정의
    dataset = MolecularDataset(csv_file=csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # 배치 크기는 필요에 따라 조절 가능
    model = AdvancedFNNModel(input_dim=dataset.features.shape[1]).to(device)  # 모델을 CUDA 장치로 이동

    # 모델을 평가 모드로 전환 (필요에 따라)
    model.eval()

    # 네트워크 층을 거친 피처를 저장할 리스트
    network_features = []

    with torch.no_grad():  # 역전파 계산을 하지 않기 위해 no_grad 사용
        for data in dataloader:
            data = data.float().to(device)  # 데이터를 float 타입으로 변환하고 CUDA 장치로 이동
            output = model(data)
            network_features.append(output.cpu().numpy())  # 결과를 numpy 배열로 변환하여 저장 (CPU로 이동)

    # 네트워크를 통과한 피처를 하나의 numpy 배열로 병합
    network_features = np.concatenate(network_features, axis=0)

    # 피처를 DataFrame으로 변환하여 반환
    features_df = pd.DataFrame(network_features)
    return features_df
