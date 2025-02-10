from os.path import join as pjoin
import os


class Config:

    def __init__(self, cnn_type="cnn-rico",version="3.8"):
        # setting CNN (graphic elements) model
        self.image_shape = (64, 64, 3)
        self.version = version

        self.project_root = '/Users/ninastidham'
        
        # For CNN trained on RICO Dataset
        if cnn_type == "cnn-rico":
            self.CNN_PATH = pjoin(self.project_root, "models", "cnn-rico-1.h5")
            if self.version != "3.8":
                self.CNN_PATH = pjoin(self.project_root, "edit_models", "converted_model.h5")

            self.element_class = ['Button', 'CheckBox', 'Chronometer', 'EditText', 'ImageButton', 'ImageView',
                              'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar', 'Spinner', 'Switch',
                              'ToggleButton', 'VideoView', 'TextView']
            self.class_number = len(self.element_class)

            self.COLOR = {'Button': (0, 255, 0), 'CheckBox': (0, 0, 255), 'Chronometer': (255, 166, 166),
                      'EditText': (255, 166, 0),
                      'ImageButton': (77, 77, 255), 'ImageView': (255, 0, 166), 'ProgressBar': (166, 0, 255),
                      'RadioButton': (166, 166, 166),
                      'RatingBar': (0, 166, 255), 'SeekBar': (0, 166, 10), 'Spinner': (50, 21, 255),
                      'Switch': (80, 166, 66), 'ToggleButton': (0, 66, 80), 'VideoView': (88, 66, 0),
                      'TextView': (169, 255, 0), 'NonText': (0,0,255),
                      'Compo':(0, 0, 255), 'Text':(169, 255, 0), 'Block':(80, 166, 66)}

            