import kagglehub

# Download latest version
path = kagglehub.dataset_download("awsaf49/brats2020-training-data") #BRATS 2020 Dataser
path = kagglehub.dataset_download("nguyenthanhkhanh/brats2024-small-dataset") #BRATS 2024 Data

print('Dataset downloaded to:', path)