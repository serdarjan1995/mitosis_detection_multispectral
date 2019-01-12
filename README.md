# mitosis_detection_multispectral
My final thesis



Project coded in Python 3.6.4
Run 'pip install -r requirements.txt' for installing necessary packages.

After installing packages simply run python script mitosis_gui.py
 - python3 mitosis_gui.py


For training dataset in provided CNN model:
model = MITOSIS_CNN(IMG_SIZE, weights=weights_path, channels=10)
model.generate_train_data(MITOSIS_PATH,NONMITOSIS_PATH)
model.train_model(epochs=30,min_delta=0.01,cross_validation=0.2)

#######################
Explanation of files in 'Code2 directory:
avg.npy   ->  discriminative vector for generating DI image
DI_generate.py    ->  generates DI image from multispectral images
mitos_model_v2.py    -> contains CNN model for predicting also for training
model_v2_test.py    -> test all images in ICPR dataset, also counts true/false positive/negative rate
save_patterns_with_mitosis_M.py    -> generate train data stacking all bands in numpy dump file
save_patterns_with_nonmitosis_M.py    -> generate train data stacking all bands in numpy dump file
weights_multispectral_v2    -> pretrained CNN model