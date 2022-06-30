# Nudenet-POC
POC of Nudenet model


    1. Replace the classifier file with following two files in ~/.NudeNet/ 
      classifier_lite.onnx
      classifier_model.onnx
    
     2. Download the following two files in ~/.NudeNet/
      detector_v2_default_checkpoint.onnx
      detector_v2_default_classes
    
     3. Delete classes file
      rm classes
      
     4. Move the detector classes file to classes
      mv detector_v2_default_classes classes 
