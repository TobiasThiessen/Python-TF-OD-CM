import os, shutil, glob, wget
from distutils.dir_util import copy_tree
import xml.etree.ElementTree as ET
from os.path import exists
import scripts.imageLabel_list
from scripts.directory_dict import directoryDict

class Workspace:
    def __init__(self):
        
        self.dirDict = directoryDict()
        
        self.dirDict.make_folderDict()

        self.labels = scripts.imageLabel_list.imagelabels
        
        if not os.path.exists(self.dirDict.folderDict["VENV_PATH"]):
            print("NO VIRTUAL ENVIRONMENT, PLEASE MAKE ONE NAMED 'VENV' AND PIP INSTALL 'requirements.txt'")
        
        if not os.path.exists(os.path.join(self.dirDict.folderDict["APIMODEL_PATH"], "research", "object_detection")):
            os.system('cmd /c "git clone https://github.com/tensorflow/models {}"'.format(self.dirDict.folderDict["APIMODEL_PATH"]))
            path0 = os.path.join(self.dirDict.folderDict["APIMODEL_PATH"], "research", "object_detection")
            path1 = os.path.join(self.dirDict.folderDict['ROOT_PATH'], "VENV", "Lib", "site-packages", "object_detection")
            copy_tree(path0, path1)

        if len(os.listdir(self.dirDict.folderDict['PROTOC_PATH'])) == 0:
            url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
            wget.download(url, out = self.dirDict.folderDict['PROTOC_PATH'])
            os.system('cmd /c "cd {} && tar -xf protoc-3.15.6-win64.zip"'.format(self.dirDict.folderDict['PROTOC_PATH']))
            os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(self.dirDict.folderDict['PROTOC_PATH'], 'bin'))   
            os.system('cmd /c "cd tensorflow/api_model/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install"')
            os.system('cmd /c "cd tensorflow/api_model/research/slim && pip install -e ."')
            print("RUNNING: 'model_builder_tf2_test.py' to verify install so far - should return 'OK' when done")
            VERIFICATION_SCRIPT = os.path.join(self.dirDict.folderDict['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
            os.system('cmd /c "python {}'.format(VERIFICATION_SCRIPT))

    def download_model(self, Model_URL, MODEL_NAME):
        PRETRAINED_MODEL_URL = Model_URL
        Splitted = PRETRAINED_MODEL_URL.split("/")
        self.PRETRAINED_MODEL_NAME = Splitted[len(Splitted) - 1].strip(".tar.gz")
        self.CUSTOM_MODEL_NAME = f"[{MODEL_NAME}]" + self.PRETRAINED_MODEL_NAME
        
        self.dirDict.update_folderDict(self.PRETRAINED_MODEL_NAME, self.CUSTOM_MODEL_NAME)
        
        # Downloads and extracts the pretrained model
        if not exists(os.path.join(self.dirDict.folderDict['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME)):
            wget.download(PRETRAINED_MODEL_URL, out = self.dirDict.folderDict['PRETRAINED_MODEL_PATH'])
            
            FileList0 = os.listdir(self.dirDict.folderDict['PRETRAINED_MODEL_PATH'])
            ArchiveDir = os.path.join(self.dirDict.folderDict["PRETRAINED_MODEL_PATH"], self.PRETRAINED_MODEL_NAME+'.tar.gz')
            shutil.unpack_archive(ArchiveDir, self.dirDict.folderDict['PRETRAINED_MODEL_PATH'], "gztar")
            FileList1 = os.listdir(self.dirDict.folderDict['PRETRAINED_MODEL_PATH'])
            
            FileName = list(set(FileList1).difference(FileList0))
            FileName = str(FileName[0])
            # Renames pretrained model to proper naming convention matching url
            if FileName != self.PRETRAINED_MODEL_NAME:
                os.chdir(self.dirDict.folderDict['PRETRAINED_MODEL_PATH'])
                os.rename(FileName, self.PRETRAINED_MODEL_NAME)

    def write_pipeline(self, BATCH_SIZE = 1):
        # Copies and edits the pipeline config file from pretrained to personal model
        if not exists(os.path.join(self.dirDict.fileDict['EDITED_PIPELINE_CONFIG'])):
            print("Copying pipeline.config")
            shutil.copyfile(self.dirDict.fileDict['UNEDITED_PIPELINE_CONFIG'], self.dirDict.fileDict['EDITED_PIPELINE_CONFIG'])
            print("Done")
            
            import tensorflow as tf
            from object_detection.utils import config_util
            from object_detection.protos import pipeline_pb2
            from google.protobuf import text_format

            print("Editing pipeline.config")
            config = config_util.get_configs_from_pipeline_file(self.dirDict.fileDict['EDITED_PIPELINE_CONFIG'])

            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(self.dirDict.fileDict['EDITED_PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
                proto_str = f.read()                                                                                                                                                                                                                                          
                text_format.Merge(proto_str, pipeline_config)  

            pipeline_config.model.ssd.num_classes = len(self.labels)
            pipeline_config.train_config.batch_size = BATCH_SIZE
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join(self.dirDict.folderDict['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path = self.dirDict.fileDict['LABELMAP']
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(self.dirDict.folderDict['ANNOTATION_PATH'], 'train.record')]
            pipeline_config.eval_input_reader[0].label_map_path = self.dirDict.fileDict['LABELMAP']
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(self.dirDict.folderDict['ANNOTATION_PATH'], 'test.record')]

            config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
            with tf.io.gfile.GFile(self.dirDict.fileDict['EDITED_PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
                f.write(config_text)
                
    # Creates record files
    def Write_Records(self):
        print("Writing labelmap")
        with open(self.dirDict.fileDict["LABELMAP"], "w") as f:
            for label in self.labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')
        # Fixes XML file, so folder & path key fits
        os.chdir(self.dirDict.folderDict['IMAGE_TRAIN_PATH'])
        for file in glob.glob("*.xml"):
            FileTree = ET.parse(file)
            FileRoot = FileTree.getroot()
            PictureName = file.split(".")
            FilePath = os.path.join(os.getcwd(), PictureName[0] + ".JPG")
            FileRoot[2].text = FilePath
            FileRoot[0].text = "train"
            FileTree.write(file)
        os.chdir(self.dirDict.folderDict['ROOT_PATH'])
        
        # Fixes XML file, so folder & path key fits
        os.chdir(self.dirDict.folderDict['IMAGE_TEST_PATH'])
        for file in glob.glob("*.xml"):
            FileTree = ET.parse(file)
            FileRoot = FileTree.getroot()
            PictureName = file.split(".")
            FilePath = os.path.join(os.getcwd(), PictureName[0] + ".JPG")
            FileRoot[2].text = FilePath
            FileRoot[0].text = "test"
            FileTree.write(file)
        os.chdir(self.dirDict.folderDict['ROOT_PATH'])
        
        os.system('cmd /c "python {} -x {} -l {} -o {} "'.format(
            self.dirDict.fileDict['TF_RECORD_SCRIPT'], self.dirDict.folderDict['IMAGE_TEST_PATH'], self.dirDict.fileDict['LABELMAP'], os.path.join(self.dirDict.folderDict['ANNOTATION_PATH'], 'test.record')))

        os.system('cmd /c "python {} -x {} -l {} -o {} "'.format(
            self.dirDict.fileDict['TF_RECORD_SCRIPT'], self.dirDict.folderDict['IMAGE_TRAIN_PATH'], self.dirDict.fileDict['LABELMAP'], os.path.join(self.dirDict.folderDict['ANNOTATION_PATH'], 'train.record')))