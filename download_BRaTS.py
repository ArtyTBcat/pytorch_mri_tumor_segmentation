import kagglehub

'''
Note!! Change default location of kagglehub download before downloading. 
I have no idea how but beware if computer storage is almost full.

'''


# Download latest version
BRaTS20 = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation") #BRATS 2020 Dataset
BRaTS24 = kagglehub.dataset_download("nguyenthanhkhanh/brats2024-small-dataset") #BRATS 2024 Dataset


print('Dataset downloaded to:', BRaTS20)
print('Dataset downloaded to:', BRaTS24)