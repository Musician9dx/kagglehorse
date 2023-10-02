import tensorflow as tf
from tensorflow.keras.layers import LSTM,Bidirectional,Dense,Input,Add,Activation,Concatenate
from tensorflow.keras import Model
import pickle
from dataclasses import dataclass
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger
import pandas as pd
import mlflow

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
        EarlyStopping(patience=8,verbose=2),
        ReduceLROnPlateau(patience=8),

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
        add=Activation("relu")(Add()([dense5,dense10]))



        dense11=Dense(1024,activation="relu")(add)
        dense12=Dense(1024,activation="tanh")(dense11)
        dense13=Dense(512,activation="relu")(dense12)
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

            history=ml.fit(x=self.x_vector,y=self.y_vector,epochs=1,callbacks=self.config.callbacks,validation_split=0.1,verbose=2,)
            
            mlflow.log_param("Start",2)
            mlflow.log_param("End",5)
            


        self.ml=ml

    def save_model(self):
        tf.keras.models.save_model(self.ml,self.config.modelSavePath+"/model.h5")


if __name__=="__main__":

    obj=ModelTrainer()
    obj.load_data()
    obj.build_ml()
    obj.train_model()

    