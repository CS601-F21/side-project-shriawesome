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

    def __init__(self,filePath):
        self.metaFile = filePath
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
        path = "../dataset/lines/"+fileName[0]+"/"+fileName[0]+"-"+fileName[1]+"/"+data+".png"
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
    Generates a dataset consisting of CombinedData objects for the data to be used for model development/ testing

    Returns
    -------
    A dictionary containing dataset along with the vocabulary
    """
    def getData(self):
        try:
            file = open(self.metaFile)
            for i,line in enumerate(file):
                if line[0]!="#":
                    lineData = line.split()
                    imgPath = self.constructImgPath(lineData[0])
                    
                    # Checks if the image size is 0
                    if not self.isBadImg(imgPath):
                        continue

                    text = (" ".join(lineData[-1].split("|")))
                    # Get the character vocabulary for a given text
                    self.__chars = self.__chars.union(list(text))
                    self.__dataset.append(CombinedData(imgPath,text))
            
            return {"dataset":self.__dataset,"Labels":sorted(self.__chars)}
        
        except FileNotFoundError:
            print(self.metaFile," : ","File not found!!!")
            
    
