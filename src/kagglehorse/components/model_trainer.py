import tensorflow as tf
from tensorflow.keras.layers import LSTM,Bidirectional,Dense,Input,Add,Activation,Concatenate
from tensorflow.keras import Model
import pickle
from dataclasses import dataclass
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger
import pandas as pd
import mlflow
from tensorflow.keras import regularizers
from kagglehorse.utils.logger import logger

@dataclass
class ModelTrainerConfig():
    
    xTrainPath="D:/INeuron/Horse/kagglehorse/resource/data/x_train.csv"
    yTrainPath="D:/INeuron/Horse/kagglehorse/resource/data/y_train.csv"
    modelSavePath="D:/INeuron/Horse/kagglehorse/resource/model"

    loss="categorical_crossentropy"
    metrics=["accuracy","loss"]
    validationRatio=0.2
    optimizer="adam"

    callbacks=[
        EarlyStopping(patience=30,verbose=1),
        ReduceLROnPlateau(patience=30),

        CSVLogger(
            "D:/INeuron/Horse/kagglehorse/resource/model/train.csv", 
                  separator=',', 
        append=False
    )]

class ModelTrainer():
    
    def __init__(self):
        self.config=ModelTrainerConfig()
    
    def load_data(self):

        self.x_vector=pd.read_csv(self.config.xTrainPath)
        self.x_vector.drop(["Unnamed: 0"],inplace=True,axis=1)
        self.y_vector=pd.read_csv(self.config.yTrainPath)
        self.y_vector.drop(["Unnamed: 0"],inplace=True,axis=1)

    
    def build_ml(self):
        inputlayer=Input((81,),name="inputlayer")

        #Parallel Branch One
        dense1=Dense(2056,activation="relu")(inputlayer)
        dense2=Dense(1024,activation="relu")(dense1)
        dense3=Dense(512,activation="relu")(dense2)
        dense4=Dense(256,activation="relu")(dense3)
        dense5=Dense(256,activation="relu")(dense4)

        #Parallel Branch One
        dense6=Dense(2056,activation="tanh")(inputlayer)
        dense7=Dense(1024,activation="relu")(dense6)
        dense8=Dense(512,activation="relu")(dense7)
        dense9=Dense(256,activation="relu")(dense8)
        dense10=Dense(256,activation="relu")(dense9)

        #Concatenate Different Outputs
        add=Activation("relu")(Concatenate()([dense5,dense10]))



        dense11=Dense(1024,activation="relu")(add)
        dropout=tf.keras.layers.Dropout(0.3)(dense11)

        dense12=Dense(1024,activation="relu")(dropout)
        dense13=Dense(512,activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5))(dense12)
        dense14=Dense(512,activation="relu")(dense13)
        output=Dense(3,activation="softmax")(dense14)
        ml=Model(inputs=inputlayer,outputs=output)

        
        self.ml=ml
        self.ml.compile(

            optimizer="adam",
            loss="categorical_crossentropy",
            metrics="accuracy"      
        )
    
    def train_model(self):

        ml=self.ml

        with mlflow.start_run():

            history=ml.fit(x=self.x_vector,y=self.y_vector,epochs=100,callbacks=self.config.callbacks,validation_split=0.1,verbose=1,)

            mlflow.log_param("Start",2)
            mlflow.log_param("End",5)
            


        self.ml=ml

    def save_model(self):


        print(self.ml.predict(self.x_vector))
        tf.keras.models.save_model(self.ml,self.config.modelSavePath+"/model.h5")


if __name__=="__main__":

    try:

        logger.info("Model Training Initialized")

        obj=ModelTrainer()
        logger.info("Load Data")
        obj.load_data()
        logger.info("Data Loading Successful")
        logger.info("Building Model")
        obj.build_ml()
        logger.info("ML Build Successful")
        logger.info("Training Model")
        obj.train_model()
        logger.info("Model Training Successful")
        obj.save_model()
        logger.info("Model Saved")
    except Exception as e:
        logger.critical(str(e))
    