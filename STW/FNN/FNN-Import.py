from FNN import extract_features_from_csv

# CSV 파일 경로를 지정하여 피처 추출
csv_file = '../../train_data/dacon_train.csv'
features_df = extract_features_from_csv(csv_file)

features_df.to_csv('output_features_2.csv', index=False)

print(features_df.shape)
print("success")
