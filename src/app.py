import cherrypy
import random
import string
import base64
import numpy as np
import cv2
import makePredictions
import os
from HTMLConstant import HTMLConstant

class HWRrender:
    @cherrypy.expose
    def index(self):
        return open("templates/landing.html")

    
    def preprocessImg(self,data,imgPath):
        img = np.asarray(bytearray(data),dtype="uint8")
        img = cv2.imdecode(img, 1)
        cv2.imwrite(imgPath,img)

    
    def generateResponse(self,text):
        header = HTMLConstant.PAGE_HEADER
        body = "<p>Predicted Text : "+text
        footer = HTMLConstant.PAGE_FOOTER
        return header + body + footer

    @cherrypy.expose
    def generate(self, image):
        imageString = base64.b64encode(image.file.read())
        imgPath = image.filename
        binary = base64.b64decode(imageString)
        self.preprocessImg(binary,imgPath)
        result = makePredictions.main(imgPath=imgPath)

        page = self.generateResponse(result)
        
        # Removing the file
        os.remove(imgPath)
    
        return page


if __name__ == '__main__':
    cherrypy.quickstart(HWRrender())
