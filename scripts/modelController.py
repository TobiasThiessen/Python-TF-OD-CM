from scripts.workspaceController import Workspace
from scripts.modelZoo_list import *
import os, os.path

def train_model(modelChoice, modelName, Training_Steps = 5000):
        
    controller = Workspace()
    controller.download_model(Model_URL = modelZoo_URLs[modelChoice], MODEL_NAME = modelName)
    controller.write_pipeline()
    controller.Write_Records()
    
    if not os.path.exists(os.path.join(controller.dirDict.folderDict["CHECKPOINT_PATH"], "eval")):
        TRAINING_SCRIPT = os.path.join(controller.dirDict.folderDict['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

        command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(TRAINING_SCRIPT, 
                    controller.dirDict.folderDict['CHECKPOINT_PATH'],controller.dirDict.fileDict['EDITED_PIPELINE_CONFIG'], Training_Steps)
        os.system(f'cmd /c "{command}"')

        # for eval to work, an exit() must be added at the end of eval_continuously in model_lib_v2 before the last return
        # otherwise it'll stay in the cmd and not continue --- make sure u moldify the correct model_lib_v2 file (most like the one in venv and not training script)
        evalcommand = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, 
                        controller.dirDict.folderDict['CHECKPOINT_PATH'],controller.dirDict.fileDict['EDITED_PIPELINE_CONFIG'], controller.dirDict.folderDict['CHECKPOINT_PATH'])
        os.system(f'cmd /c "{evalcommand}"')
    else:
        print(f"This model has already been trained and evaluated")

def model_trainResults(modelChoice = 0):
    controller = Workspace(0, 0, modelZoo_URLs[modelChoice])
    results = "tensorboard --logdir={}\.".format(controller.dirDict.folderDict['TRAINING_RESULTS_TRAIN_PATH'])
    os.system(f'cmd /c "{results}"')
    
def model_evalResults(modelChoice = 0):
    controller = Workspace(0, 0, modelZoo_URLs[modelChoice])
    results = "tensorboard --logdir={}\.".format(controller.dirDict.folderDict['TRAINING_RESULTS_EVAL_PATH'])
    os.system(f'cmd /c "{results}"')
    
def model_graphFreeze(Training_Steps = 0, modelChoice = 0):
    PictureAmountTrain = int(len(os.listdir('.\/tensorflow/images/train')) / 2)
    controller = Workspace(PictureAmountTrain, Training_Steps, modelZoo_URLs[modelChoice])
    FREEZE_SCRIPT = os.path.join(controller.dirDict.folderDict['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py')
    command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,controller.dirDict.fileDict['EDITED_PIPELINE_CONFIG'], controller.dirDict.folderDict['CHECKPOINT_PATH'], controller.dirDict.folderDict['OUTPUT_PATH'])
    os.system(f'cmd /c "{command}"')
    
"""
def unused():
        trainingFile = os.listdir(os.path.join(controller.dirDict.folderDict["CHECKPOINT_PATH"], "train"))
        trainingFileDir = os.path.join(controller.dirDict.folderDict["CHECKPOINT_PATH"], "train", trainingFile[0])
        shutil.copy(trainingFileDir, controller.dirDict.folderDict['TRAINING_RESULTS_TRAIN_PATH'])
        
        evalFile = os.listdir(os.path.join(controller.dirDict.folderDict["CHECKPOINT_PATH"], "eval"))
        evalFileDir = os.path.join(controller.dirDict.folderDict["CHECKPOINT_PATH"], "eval", evalFile[0])
        shutil.copy(evalFileDir, controller.dirDict.folderDict['TRAINING_RESULTS_EVAL_PATH'])
"""