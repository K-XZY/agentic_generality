# Agentic Generality
A repository on the research and engineering of general purpose GUI agent.

It will host code from the UCL MEng RAI Industry Project. As well as other work members did in AI.

## Taking Screenshots using Android Env 


### Access the correct directory

```cd ari_perceiver```

### Run screenshot code via the terminal 

```python screenshot.py --task_path 'temp.textproto'```

the above specified can be ran using a python version > 3.10. Screenshots to be saved to ```input``` file under the name of ```screenshot.png```, each time this overrides the pre-existing ```screenshot.png``` file 

## Running CNN and detection model

### Creating Anaconda Python Environment (Optional)

CNN performance overall shown to be optimal when ran within a environment in which the python version is 3.8

```conda create -n legacy_env python=3.5```
```conda activate legacy_env```

Following environment setup ensure to install compatible versions of both keras and tensorflow with python 3.8

### Don't want to use python 3.8

in ```run_single.py``` line 95, change version from "3.8" to "3.10" and run as usual. Python version 3.8 is default for the code 

### Running UI detection on  Single Screenshot

```python run_single.py```

A series of images will display starting with text extracted, followed by bounding box of UI elements motivated by ```cv2```, CNN will be applied to show bounding boxes of the different types of UI e.g. TextView,ImageButton etc. (see ReDraw dataset for all categories), hold down key 0 to transition between different stages

```output file``` - contains jsons at each step with merge folder containing the final json for the component text coordinates and classification provided by the CNN. ```output/merge/screenshot.json``` is the file for parsing into the LLM for action sequence generation.

### Final Notes 

```CONFIG.py``` specifies model paths for versions before and after python 3.10. Not device specific, general use. 







