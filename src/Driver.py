import DataLoader
from ModelConfig import ModelConfig

class Driver:

    def saveFile(fileName,data):
        with open("../"+fileName,"w") as file:
            file.write(data)

    if __name__=="__main__":
        # Generate dataset
        dataLoader = DataLoader.DataLoader(ModelConfig.DATA_CONFIG_FILE,ModelConfig.MAX_TEXT_LEN)
        dataSet = dataLoader.getData()

        # Save character list to the file
        saveFile("charList.txt",''.join(dataSet["Labels"]))
