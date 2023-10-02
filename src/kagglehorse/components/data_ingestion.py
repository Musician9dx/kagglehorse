from kagglehorse.utils.logger import logger
import os
import pandas as pd
import numpy as np
import pickle
from pymongo.mongo_client import MongoClient
from dataclasses import dataclass
"""try:
    mongo=MongoClient("mongodb+srv://mymongo:mymongo@cluster0.paomhx1.mongodb.net/")
    db=mongo["horse"]
    clc=db["horse"]
    data=clc.find({})

    for i in data:
        print(i)
except Exception as e:
    print(str(e))"""

@dataclass
class DataIngestionConfig():
    connectionUrl="mongodb+srv://mymongo:mymongo@cluster0.paomhx1.mongodb.net/"
    dataDestination="D:/INeuron/Horse/kagglehorse/resource/data"
    databaseName="horse"
    collectionName="horse"

class DataIngestion():
    
    def __init__(self):
        self.config=DataIngestionConfig()
    
    def connect_to_pymongo(self):
        mongo=MongoClient("mongodb+srv://mymongo:mymongo@cluster0.paomhx1.mongodb.net/")
        db=mongo["horse"]
        clc=db["horse"]
        self.cursor=clc
    
    def fetch_data(self):
        document=[]

        for i in self.cursor.find({}):
            document.append(i)
        
        self.data=pd.DataFrame(document)

    def save_data(self):

        data=self.data
        
        data.to_csv("D:/INeuron/Horse/kagglehorse/resource/data/train.csv")

        print(data)
        
    
if __name__=="__main__":

    try:
        logger.info("Data Ingestion Initialized")

        obj=DataIngestion()

        logger.info("Read Config")
        logger.info("Connecting to Pymongo")

        obj.connect_to_pymongo()

        logger.info("Pymongo Cursor Returned")
        logger.info("Fetching Data")

        obj.fetch_data()
        logger.info("Data Fetch successful")
        logger.info("Save Data")

        obj.save_data()
        logger.info("Data Saved")
        logger.info("Data Ingestion Terminated")



    except Exception as e:
        logger.critical(str(e))


   

