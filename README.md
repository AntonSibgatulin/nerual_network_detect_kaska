# Helmet, Persone, Glasses detection Python 3.8.10

```shell
pip install pyqt5
pip install lxml
pip install pillow
pip install numpy
pip install matplotlib
pip install opencv-python
pip install tensorflow
pip install keras
pip install contextlib2
pip install pandas
pip install Cython
pip install jupyter
```

```shell
mkdir workdir/
cd workdir
git clone https://github.com/EscVM/OIDv4_ToolKit.git
cd OIDv4_ToolKit
python -m pip install -r requirements.txt
python -m pip install lxml
```

Then you should check classes, you ccan find it on website:
https://storage.googleapis.com/openimages/web/visualizer/index.html


```shell
python main.py downloader --classes {SOME CLASSES} --type_csv train --limit {COUNT IMAGES}
```

run (also should add cellphone)

```shell
python main.py downloader --classes Helmet Person Glasses --type_csv train --limit 1000
```

change file classes.txt

```shell
nano classes.txt
```

```text
Helmet
Glasses
Person
```
then `CNTRL+X` and then yes


```shell
nano convert_annotations.py
```

```python
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import fileinput

# function that turns XMin, YMin, XMax, YMax coordinates to normalized yolo format
def convert(filename_str, coords):
    os.chdir("..")
    image = cv2.imread(filename_str + ".jpg")
    coords[2] -= coords[0]
    coords[3] -= coords[1]
    x_diff = int(coords[2]/2)
    y_diff = int(coords[3]/2)
    coords[0] = coords[0]+x_diff
    coords[1] = coords[1]+y_diff
    coords[0] /= int(image.shape[1])
    coords[1] /= int(image.shape[0])
    coords[2] /= int(image.shape[1])
    coords[3] /= int(image.shape[0])
    os.chdir("Label")
    return coords

ROOT_DIR = os.getcwd()

# create dict to map class names to numbers for yolo
classes = {}
with open("classes.txt", "r") as myFile:
    for num, line in enumerate(myFile, 0):
        line = line.rstrip("\n")
        classes[line] = num
    myFile.close()
# step into dataset directory
os.chdir(os.path.join("OID", "Dataset"))
DIRS = os.listdir(os.getcwd())

# for all train, validation and test folders
for DIR in DIRS:
    if os.path.isdir(DIR):
        os.chdir(DIR)
        print("Currently in subdirectory:", DIR)
        
        CLASS_DIRS = os.listdir(os.getcwd())
        # for all class folders step into directory to change annotations
        for CLASS_DIR in CLASS_DIRS:
            if os.path.isdir(CLASS_DIR):
                os.chdir(CLASS_DIR)
                print("Converting annotations for class: ", CLASS_DIR)
                
                # Step into Label folder where annotations are generated
                os.chdir("Label")

                for filename in tqdm(os.listdir(os.getcwd())):
                    filename_str = str.split(filename, ".")[0]
                    if filename.endswith(".txt"):
                        annotations = []
                        with open(filename) as f:
                            for line in f:
                                for class_type in classes:
                                    line = line.replace(class_type, str(classes.get(class_type)))
                                labels = line.split()
                                coords = np.asarray([float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])])
                                coords = convert(filename_str, coords)
                                labels[1], labels[2], labels[3], labels[4] = coords[0], coords[1], coords[2], coords[3]
                                newline = str(labels[0]) + " " + str(labels[1]) + " " + str(labels[2]) + " " + str(labels[3]) + " " + str(labels[4])
                                line = line.replace(line, newline)
                                annotations.append(line)
                            f.close()
                        os.chdir("..")
                        with open(filename, "w") as outfile:
                            for line in annotations:
                                outfile.write(line)
                                outfile.write("\n")
                            outfile.close()
                        os.chdir("Label")
                os.chdir("..")
                os.chdir("..")
        os.chdir("..")
```

```shell
python convert_annotations.py
```


```shell
nano generator.py
```

```python
import xml.etree.ElementTree as ET
from os import getcwd
import os

dataset_train = 'OID/Dataset/train/'

dataset_file = 'train.txt'
classes_file = 'obj.names'

CLS = os.listdir(dataset_train)
classes = [dataset_train + CLASS for CLASS in CLS]
wd = getcwd()


def test(fullname):
    list_file = open(dataset_file, 'a')
    file_string = str(fullname)[:-4] + '.jpg\n'
    list_file.write(file_string)
    list_file.close()


for CLASS in classes:
    for filename in os.listdir(CLASS):
        if not filename.endswith('.txt'):
            continue
        fullname = os.getcwd() + '/' + CLASS + '/' + filename
        test(fullname)

for CLASS in CLS:
    list_file = open(classes_file, 'a')
    file_string = str(CLASS) + "\n"
    list_file.write(file_string)
    list_file.close()

```

```shell
python generator.py
```



```shell
cd ..
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
```

Откройте Makefile и измените след переменные на
```shell
GPU = 1
CUDA = 1
HALF_CUDA = 1
OPENCV = 1
```

```shell
make
```

### After Downloading


```shell
mkdir dt
cp cfg/yolov3.cfg dt/yolov3.cfg
```

Comment below lines(Line: 2 to 5):
# Testing
# batch=1
# subdivisions=1
# Training
Uncomment(Line: 6 and 7):
batch=64
subdivisions=16

Now, go to subdivisions (i.e., Line: 7), replace according to ur GPUs.
If subdivisions=16 and gives CUDA error while training, then change it to 32 ie, subdivisions=32, if still it gives error then change to 64 and so on.

Now, search for max_batches on (Line:20). Change it accordingly:
it depends on number of your classes.
max_batches = classes * 2000
→ If you have 3 classes, then max_batches = 6000 (3*2000)
→If you have 4 classes, then max_batches = 8000 (4*2000)
→If you have 1 classes, then max_batches = 4000. (4000 because, minimum max_batches should be set at 4000 for better training)

Now, just below max_batches, steps are there, it also be changed accordingly:
if max_batches = 6000 then
steps = 4800, 5400

Now, search for yolo. And u will [yolo]. In that, you will find classes:80.
Change this classes to number of your classes. Eg, classes = 4

And above this [yolo] layer, you will find a [conventional] layer. In this, you will find a filters attribute:255. Change it accordingly.
filters = (classes+5)*3
→ If classes = 4; then filters = 27 (4+5)*3
→ If classes = 1; then filters = 18 (1+5)*3

Now, this we have to modify all [yolo] and [conventional] layer. So just repeat this steps. (it will be repeated 3 times only).

After this, one small thing is left, that is, random. Search for random, and change accordingly.
random = 1 → It resizes images while training and doesnow overfit.(I prefer this.)
random = 0 → it will not resizes and No problem of out of memory. (can be risky)


```shell
cd ..
cp OIDv4_ToolKit/train.txt darknet/dt/test.txt
cp OIDv4_ToolKit/train.txt darknet/dt/train.txt
cp OIDv4_ToolKit/obj.names darknet/dt/obj.names
```

```shell
cd darknet/dt
nano obj.data
```

```editorconfig
classes = 3
train = dt/train.txt
valid = dt/test.txt
names = dt/obj.names
backup = backup/
```

```shell
sudo apt-get install cmake
sudo apt-get install gcc g++
# python2
sudo apt-get install python-dev python-numpy
# python3
sudo apt-get install python3-dev python3-numpy

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev


sudo apt-get install libgtk2.0-dev


sudo apt-get install libgtk-3-dev

sudo apt-get install libpng-dev
sudo apt-get install libjpeg-dev
sudo apt-get install libopenexr-dev
sudo apt-get install libtiff-dev
sudo apt-get install libwebp-dev

git clone https://github.com/opencv/opencv.git

mkdir build
cd build

sudo apt install cmake

cmake ../


make
sudo make install


```



```shell
cd ..
wget https://pjreddie.com/media/files/darknet53.conv.74
```

Run train our model

```shell
./darknet detector train dt/obj.data dt/yolov3.cfg darknet53.conv.74 -dont_show -i 0 -map -gpus 0
```



