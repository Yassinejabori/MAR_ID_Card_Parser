#import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import cv2
import pytesseract
import re
import os
import numpy as np
import scanner
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'

UPLOAD_FOLDER='static'

app = Flask(__name__)

@app.route('/cin_front',methods=['POST'])
def front_predict():
    if request.method == "POST" :
        image_file=request.files['']
        if image_file:
            
            image_location=os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
            
            scanner.scan(image_location, "front")
            img = cv2.imread("temp_front.jpg", cv2.IMREAD_GRAYSCALE)

            im_bw = img
            thresh = 120
            im_bwar = cv2.threshold(im_bw, thresh + 30 , 255, cv2.THRESH_BINARY)[1]
            im_bw = cv2.threshold(im_bw, thresh, 255, cv2.THRESH_BINARY)[1]

            cin = im_bw[504:544, 715:920]
            first_name = im_bw[198:246, 14:390]
            last_name = im_bw[282:328, 14:380]
            naissance = im_bw[324:365, 221:430]
            expiration = im_bw[442:490, 290:490]
            lieu_naiss = im_bw[410:450, 40:360]
            arab_prenom = im_bwar[159:210, 346:690]
            arab_nom = im_bwar[245:300, 372:700]
            arab_lieu = im_bwar[360:410, 339:650]

            arab_lieu =pytesseract.image_to_string(arab_lieu,lang="ara")
            arab_nom =pytesseract.image_to_string(arab_nom,lang="ara")
            arab_prenom =pytesseract.image_to_string(arab_prenom,lang="ara")
            lieu_naiss =pytesseract.image_to_string(lieu_naiss,lang="fra")
            expiration =pytesseract.image_to_string(expiration,lang="fra")
            naissance =pytesseract.image_to_string(naissance,lang="fra")
            cin_id =pytesseract.image_to_string(cin,lang="fra")
            last_name =pytesseract.image_to_string(last_name,lang="fra")
            first_name=pytesseract.image_to_string(first_name,lang="fra")

        return jsonify({'CIN': cin_id,'First Name':first_name,'Last name': last_name,'Naissance':naissance,'Arab name': arab_nom,'Arab Prenom':arab_prenom,'Arab lieu':arab_lieu, 'Expiration':expiration}),200
    return jsonify({'OUTPUT': 'bad request'}),400 

@app.route('/cin_back',methods=['POST'])
def back_predict():
    if request.method == "POST" :
        image_file=request.files['']
        if image_file:
            
            image_location=os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
            
            scanner.scan(image_location, "back")
            img = cv2.imread("temp_back.jpg", cv2.IMREAD_GRAYSCALE)

            im_bw = img
            thresh = 120
            im_bwar = cv2.threshold(im_bw, thresh + 30 , 255, cv2.THRESH_BINARY)[1]
            im_bwp = cv2.threshold(im_bw, thresh + 10, 255, cv2.THRESH_BINARY)[1]
            im_bw = cv2.threshold(im_bw, thresh + 20, 255, cv2.THRESH_BINARY)[1]
            
            address = im_bw[303:370, 130:980]
            #sexe = im_gray[374:420, 800:876]
            sexe = im_bw[375:430, 691:991]
            #test = im_bw[y1:y2,x1:x2]
            parents = im_bwp[135:220, 13:633]
            arab_parents = im_bwar[57:145, 479:990]
            arab_address = im_bwar[215:300, 87:910]

            arab_address = pytesseract.image_to_string(arab_address,lang="ara")
            arab_parents = pytesseract.image_to_string(arab_parents,lang="ara")
            parents = pytesseract.image_to_string(parents,lang="fra")
            sexe = pytesseract.image_to_string(sexe,lang="eng+ara")
            address=pytesseract.image_to_string(address,lang="fra")

        return jsonify({'Arab Address': arab_address,'Arab Parents':arab_parents,'Address': address,'Parents':parents,'Sexe': sexe}),200
    return jsonify({'OUTPUT': 'bad request'}),400

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
       "please wait until server has fully started"))

    app.run(debug=True, port=8080)