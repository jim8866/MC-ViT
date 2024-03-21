# MC-ViT
Software: Python 3.7.0 Pytorch 1.10.0

Usage Guide:

This package include the 3D Patch Network code and 64 datasets for testing.

Download all the zip files and extract all.

Extract features from 3D image by 3D Patch Network.

 #3DCNN/main.py for training the 3D data
 
         Input data: 
	 
                 3D images data/Patch_demo_ADNI_3Ddata
 #3DCNN/main_0422_get_data.py for extracting the 3D Patch MRI and evaluate the BA_ACC for each patch
 
        Input data: 
	
	    3D images data/Patch_demo_ADNI_3Ddata
     
        Output data:
	
            adni_gvcnn_train2_cn_ad_D_patch_400_label.mat 
	    
            result/result~~~/fold-0/cnn_labels/section/cn_ad_label_test_slice.csv  for evaluating the BA_ACC for each patch
	
.P_Vote : Accuray, fscore, gmean, sensitivity and specificity  
