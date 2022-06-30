""" Simple Utility to test Nudenet """

import os
import sys
import json
import numpy as np
import glob

import warnings
warnings.filterwarnings("ignore")

""" Import nudenet module """
from nudenet import NudeClassifier
from nudenet import NudeDetector


# Path to root folder of the app
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to configuration folder in root directory
APP_INPUT = os.path.join(APP_ROOT, 'input')

# Path to keep static files
APP_OUTPUT = os.path.join(APP_ROOT, 'output')


def myConverter(o):
    if isinstance(o, np.float32):
        return float(o)


def initNudeNet():
    # initialize classifier (downloads the checkpoint file automatically the first time)
    """ https://kandi.openweaver.com/python/notAI-tech/NudeNet#Community-Discussions """
    myVideoClassifier = NudeClassifier()
    print("NudeNet classifier loaded!")

    # initialize detector (downloads the checkpoint file automatically the first time)
    """ https://pastebin.com/rkGP0JhD """
    """ 
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
    """
    myLabelDetector = NudeDetector()
    print("NudeNet detector loaded!")
    nudeNetStatus = True

    return nudeNetStatus, myVideoClassifier, myLabelDetector


def createDir():
    """ Create the input directory """
    if not os.path.exists(APP_INPUT):
        os.mkdir(APP_INPUT)

    """ Create the output directory """
    if not os.path.exists(APP_OUTPUT):
        os.mkdir(APP_OUTPUT)


if __name__ == '__main__':

    """ Create input and output directories if not created already """
    createDir()

    """ Initialize the Nudenet classifier and Detector """
    status, myVideoClassifier, myLabelDetector = initNudeNet()
    if not status:
        print("Nudenet fails to initialize so exiting....")
        sys.exit()
    else:
        """ Read all files from APP_INPUT and run this model """
        for file in glob.glob(os.path.join(APP_INPUT, '*')):
            filename = file.split('/')[-1]
            print(f'Processing for {filename}')

            """" Extract the first name """
            filename = file.split('/')[-1].split('.')[0]
            """ Call NudeNet v2 classifier to get prediction score """
            classifierResult = myVideoClassifier.classify_video(file)
            print("****************************************")
            print(f'Classification Result {classifierResult}')
            print("****************************************")

            """ Dump the result in json file """
            outfile = filename+'_classifier'+'.'+'json'
            with open((os.path.join(APP_OUTPUT, outfile)), "w") as write_file:
                json.dump(
                    classifierResult, write_file, default=myConverter, sort_keys=True, indent=4, separators=(',', ': '))

            print(" ")

            """ Call Nudenet v2 detector to get per frame score and labels """
            detectorResult = myLabelDetector.detect_video(file, show_progress=True)
            print("****************************************")
            print(f'Detector Result {detectorResult}')
            print("****************************************")

            """ Dump the result in json file """
            outfile = filename + '_detector' + '.' + 'json'
            with open((os.path.join(APP_OUTPUT, outfile)), "w") as write_file:
                json.dump(detectorResult, write_file, default=myConverter, sort_keys=True, indent=4, separators=(',', ': '))