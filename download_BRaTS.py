import kagglehub

'''
Note!! Change default location of kagglehub download before downloading. 
I have no idea how but beware if computer storage is almost full.

'''


# Download latest version
path = kagglehub.dataset_download("awsaf49/brats2020-training-data", path='') #BRATS 2020 Dataser
path = kagglehub.dataset_download("nguyenthanhkhanh/brats2024-small-dataset") #BRATS 2024 Data

print('Dataset downloaded to:', path)