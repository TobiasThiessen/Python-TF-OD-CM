import os

class directoryDict:
    def __init__(self):

        self.TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
        self.LABEL_MAP_NAME = "label_map.pbtxt"

        # Gets root folder - very bad method, only works if this file is in a directory named "scripts" 1 up from root.
        FolderList = os.getcwd().split("\\")
        RootPath = ""
        for i in range (0, (len(FolderList))):
            RootPath = RootPath + FolderList[i] + "\\"
            
        os.chdir(RootPath)       
        
        self.folderDict = {
            "ROOT_PATH": RootPath,
            "VENV_PATH": os.path.join(                  "VENV"),
            "SCRIPTS_PATH": os.path.join(               "scripts"),
            "WORKSPACE_PATH": os.path.join(             "tensorflow"),
            "ANNOTATION_PATH": os.path.join(            "tensorflow", "annotations"),
            "APIMODEL_PATH": os.path.join(              "tensorflow", "api_model"),
            "IMAGE_PATH": os.path.join(                 "tensorflow", "images"),
            "IMAGE_TEST_PATH": os.path.join(            "tensorflow", "images", "test"),
            "IMAGE_TRAIN_PATH": os.path.join(           "tensorflow", "images", "train"),
            "MODEL_PATH": os.path.join(                 "tensorflow", "models"),
            "TRAINING_RESULTS_PATH": os.path.join(      "tensorflow", "models", "results"),
            "PROTOC_PATH": os.path.join(                "tensorflow", "protoc"),
            "PRETRAINED_MODEL_PATH": os.path.join(      "tensorflow", "pretrained_models"),
            "SELF_TRAINED_MODEL_PATH":os.path.join(     "tensorflow", "selftrained_models"),
            "EXPORT_PATH": os.path.join(                "tensorflow", "selftrained_models", "export"),
            "TFJS_PATH": os.path.join(                  "tensorflow", "selftrained_models", "TFJS-export"),
            "TFLITE_PATH": os.path.join(                "tensorflow", "selftrained_models", "TFLITE-export")
        }

        self.fileDict = {
            "TF_RECORD_SCRIPT": os.path.join(self.folderDict["SCRIPTS_PATH"], self.TF_RECORD_SCRIPT_NAME),
            "LABELMAP": os.path.join(self.folderDict["ANNOTATION_PATH"], self.LABEL_MAP_NAME)
        }
    
    def update_folderDict(self, PRETRAINED_MODEL_NAME, CUSTOM_MODEL_NAME):
        self.folderDict.update({
            "TRAINING_RESULTS_MODEL_PATH": os.path.join("tensorflow", "models", "results", "[results]" + PRETRAINED_MODEL_NAME),
            "TRAINING_RESULTS_TRAIN_PATH": os.path.join("tensorflow", "models", "results", "[results]" + PRETRAINED_MODEL_NAME, "train"),
            "TRAINING_RESULTS_EVAL_PATH": os.path.join( "tensorflow", "models", "results", "[results]" + PRETRAINED_MODEL_NAME, "eval"),
            "CHECKPOINT_PATH": os.path.join(            "tensorflow", "models", CUSTOM_MODEL_NAME),
            "EXPORT_PATH": os.path.join(                "tensorflow", "selftrained_models", "export", CUSTOM_MODEL_NAME),
            "TFJS_PATH": os.path.join(                  "tensorflow", "selftrained_models", "TFJS-export", CUSTOM_MODEL_NAME),
            "TFLITE_PATH": os.path.join(                "tensorflow", "selftrained_models", "TFLITE-export", CUSTOM_MODEL_NAME)
        })
        
        self.fileDict.update({
            "UNEDITED_PIPELINE_CONFIG": os.path.join(   self.folderDict["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "pipeline.config"),
            "EDITED_PIPELINE_CONFIG": os.path.join(     self.folderDict["MODEL_PATH"], CUSTOM_MODEL_NAME, "pipeline.config"),
            "TRAINING_RESULTS_TRAIN": os.path.join(     self.folderDict["MODEL_PATH"], "results", "train", "trainResults" + CUSTOM_MODEL_NAME),
            "TRAINING_RESULTS_EVAL": os.path.join(      self.folderDict["MODEL_PATH"], "results", "eval", "evalResults" + CUSTOM_MODEL_NAME)
        })
        
        self.make_folderDict()
        
    def make_folderDict(self):
        for folder in self.folderDict.values():
            if not os.path.exists(folder):
                if folder == "VENV":
                    pass
                else:
                    print("Making", folder)
                    os.mkdir(folder)
                    print("Done")