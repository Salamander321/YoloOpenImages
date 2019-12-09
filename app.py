import os
import darknet
from flask import Flask,abort, jsonify, render_template, request 
import tempfile
import cv2
import time
import numpy as np
import tempfile

app = Flask(__name__)



#creating Yolo Object
netMain = None
metaMain = None
altNames = None


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections):
    _class,_score,_box = [],[],[]
    for detection in detections:
        item,score,boxes = detection
        x, y, w, h = boxes
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        _class.append(item.decode())
        _score.append(score)
        _box.append([ymin,xmin,ymax,xmax])
    return _class,_score,_box


    
def YOLO():
    global netMain, metaMain, altNames
    configPath = "./shirt/yoloShirt.cfg"
    weightPath = "./shirt/yolo-obj_1700.weights"
    metaPath = "./openImage.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"),0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    print(altNames)



@app.route('/detect',methods = ['POST','GET'])
def detectImage():
    start = time.time()
    if request.method == "GET":
        print("Have got Get request this is for debug")
        return None
    #For Post Method
    tf = tempfile.NamedTemporaryFile()
    filename = str(tf.name) + ".jpg"
    tf.close()
    
    image = request.files.get("image_file")
    image.save(filename)
    tempImage = cv2.imread(filename)
    h,w,c = tempImage.shape
    darknet_image = darknet.make_image(w,h,3)
    darknet.copy_image_from_bytes(darknet_image,tempImage.tobytes())
    detection = darknet.detect_image(netMain, metaMain,darknet_image, thresh=0.25)
    print("Total Time is {}".format(time.time()-start))
    os.remove(filename)
    _it,_sc,_box = cvDrawBoxes(detection)
    return jsonify( data = [_it,_sc,_box])
    


if __name__ == "__main__":
    YOLO()
    app.run(host="0.0.0.0", port=8080,debug=True)

