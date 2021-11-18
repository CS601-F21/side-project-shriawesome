class DataLoader:
    


    def __init__(self,filePath):
        self.metaFile = filePath
        self.imgPaths = []
        self.texts = []

    def constructImgPath(self,data):
        print(data)

    def getData(self):
        try:
            file = open(self.metaFile)
            for i,line in enumerate(file):
                if line[0]!="#":
                    lineData = line.split()
                    imgPath = self.constructImgPath(lineData[0])
                    print(line)

                    if i==30:
                        break
                
            


        except FileNotFoundError:
            print(self.metaFile," : ","File not found!!!")
            
    



if __name__=="__main__":
    dataLoad = DataLoader("../dataset/metadata/lines.txt")
    dataLoad.getData()
