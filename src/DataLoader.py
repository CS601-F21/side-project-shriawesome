import os

class CombinedData:
    """
    Helper Class for DataLoader to store 2 values to be consumed for building model
    """
    def __init__(self,path,text):
        self.imgPath = path
        self.text = text 


class DataLoader:
    """ 
    Helper class for the main class to be used for generating a dataset to be used for model development
    """

    def __init__(self,filePath,maxTextLen):
        self.metaFile = filePath
        self.maxTextLen = maxTextLen
        self.__dataset = []
        self.__chars = set()

    """
    Constructs a path that contains a location for the images to be used for model development/ testing
    Parameters
    -----------
    data : raw file name to be converted to the exact file location

    Returns 
    --------
    Path converted string 
    """
    def constructImgPath(self,data):
        fileName = data.split("-")
        path = "../dataset/words/"+fileName[0]+"/"+fileName[0]+"-"+fileName[1]+"/"+data+".png"
        return path


    """
    Check if the size of the input.

    Parameters
    ----------
    imgPath : path of the image file

    Returns
    -------
    Size of the input image
    """
    def isBadImg(self,imgPath):
        return os.path.getsize(imgPath)


    """
    Limits the size of the text to the maximum text length.

    Parameters
    -----------
    - text : raw text
    - maxTextLen : Maximum text size

    Returns
    -------
    String within the maxTextLen

    """
    def processText(self,text,maxTextLen):
        # If a label is very long then the ctc-loss returns an infinite gradient
        # Repeated Letters costs double because of the blank symbol needed to be inserted
        cost = 0
        for i in range (len(text)):
            if i!=0 and text[i]==text[i-1]:
                cost+=2
            else:
                cost+=1

            if cost>=maxTextLen:
                text = text[:i]
                break

        return text


    """
    Generates a dataset consisting of CombinedData objects for the data to be used for model development/ testing

    Returns
    -------
    A dictionary containing dataset along with the vocabulary
    """
    def getData(self):
        try:
            with open(self.metaFile) as file:
                lineCount=0
                maxLabel=0
                for i,line in enumerate(file):
                    if line[0]!="#" and line.split(" ")[1]!="err":
                        lineData = line.strip()
                        lineData = line.split(" ")
                        imgPath = self.constructImgPath(lineData[0])

                        # Checks if the image size is 0
                        if not self.isBadImg(imgPath):
                            continue
                    
                        text = ("".join(lineData[-1].strip()))
                        labels = text.split(" ")
                        if len(text)>maxLabel:
                            maxLabel=len(text)
                        # Get the character vocabulary for a given text
                        # text = self.processText(text.strip(),self.maxTextLen)
                        text = text.strip()
                        self.__chars = self.__chars.union(list(text))
                        self.__dataset.append(CombinedData(imgPath,text))
            
            return {"dataset":self.__dataset,"Labels":sorted(self.__chars)}
        
        except FileNotFoundError:
            print(self.metaFile," : ","File not found!!!")
        

    def loadCharSet(self):
        with open('../charList.txt') as file:
            data = file.readlines()
        
        return data[0]