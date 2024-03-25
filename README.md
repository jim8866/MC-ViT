# MC-ViT
Software: Python 3.7.0 Pytorch 1.10.0

Usage Guide:

This package include the 3D Patch Network code and a demo data from ADNI for testing. 
(To safeguard subject privacy, we have concealed all sensitive data for the demo data.)

Download all the zip files and extract all.

Extract features from 3D image by 3D Patch Network.

 #3DCNN/main_0422_get_data.py for extracting the 3D Patch MRI
 
        Input data: 
	
	    data/demo_demo_demo.nii.gz (Sample data from the test set)
     	    data/demo.csv
     
        Output data:
	
            adni_gvcnn_train2_cn_ad_D_patch_400_label.mat 
	    result_3d_cnn\result\fold-0\cnn_classification\new_selection\test_mode_cv5_level_metrics.csv (result for evaluate the test data, including Accuray, fscore, gmean, sensitivity and specificity)
            result_3d_cnn\result\fold-0\cnn_labels\selection   (Predictions for each patch within every scan)
            
