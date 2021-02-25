# OL-VAE
### Dataset
* Add KDD99 dataset to the `dataset/KDD99/raw/`. training data: kddcup.data_10_percent_corrected, testing data: corrected.
* If you are the first time to run the project, you need make the parameter `preprocessed=True` in `kdd99_train_loader` or `kdd99_test_loader` to get the processed training or testing data. If you are not the first to run, `preprocessed=False`.
* After getting the processed data, there will be 8 files in `dataset/KDD99/raw/` named `kddcup.data_10_percent_corrected`(the original data file),`kddcup.data_10_percent_corrected_number.cvs`, `kddcup.data_10_percent_corrected_feature.cvs` and `kddcup.data_10_percent_corrected_destination.cvs` for training data, and `corrected`(the original test file),`corrected_number.cvs`, `corrected_feature.cvs` and `corrected_destination.cvs` for testing data. The `kddcup.data_10_percent_corrected_destination.cvs` is the final training data, and the `corrected_destination.cvs` is th final testing data.
