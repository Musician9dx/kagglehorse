from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kagglehorse.utils.logger import logger
import scipy
import pandas as pd 
import numpy as np
from dataclasses import dataclass
import pickle

@dataclass
class DataTransformationConfig():

    trainDataPath="D:/INeuron/Horse/kagglehorse/resource/data/train.csv"
    transformerDataDestinationPath="D:/INeuron/Horse/kagglehorse/resource/data"

    data_columns=['id', 'surgery', 'age', 'hospital_number', 'rectal_temp', 'pulse',
       'respiratory_rate', 'temp_of_extremities', 'peripheral_pulse',
       'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis',
       'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux',
       'nasogastric_reflux_ph', 'rectal_exam_feces', 'abdomen',
       'packed_cell_volume', 'total_protein', 'abdomo_appearance',
       'abdomo_protein', 'surgical_lesion', 'lesion_1', 'lesion_2', 'lesion_3',
       'cp_data', 'outcome']
    
    numericalVariables=[ 'rectal_temp', 'pulse','respiratory_rate',"pulse",'respiratory_rate', "nasogastric_reflux_ph",'packed_cell_volume', 'total_protein','abdomo_protein',"lesion_1"]

    categoricalVariables=["abdomo_appearance","pain","capillary_refill_time","mucous_membrane","abdomen","rectal_exam_feces","nasogastric_reflux","nasogastric_tube","abdominal_distention","peristalsis","pain",'temp_of_extremities', 'peripheral_pulse',]

    targetVariable=["outcome"]

    dropVariables=["id","hospital_number","_id"]


class DataTransformation():
    
    def __init__(self) -> None:
        self.config=DataTransformationConfig()

    def load_data(self):
        
        self.data=pd.read_csv(self.config.trainDataPath)
        print(self.data)
    
    def drop_columns(self):

        self.data.drop(self.config.dropVariables,axis=1,inplace=True)
    
    def buildTransformer(self):

        numericalPipeline=Pipeline([

            ("Imputer",SimpleImputer(strategy="median")),
            ("StandardScaler",StandardScaler())

        ])

        categoricalPipeline=Pipeline([

            ("Imputer",SimpleImputer(strategy="most_frequent")),
            ("OneHotEncoder",OneHotEncoder())

        ])
        Transformer=ColumnTransformer([

        ("NumericalVariables",numericalPipeline,self.config.numericalVariables),
        ("CategoricalVariables",categoricalPipeline,self.config.categoricalVariables)

        ])

        file=open("D:/INeuron/Horse/kagglehorse/resource/model/transformer.pkl","wb")
        pickle.dump(Transformer,file)




        self.Transformer=Transformer
    
    def transform_data(self):



        self.data["cp_data"].replace({
            "yes":1,
            "no":0}
        ,inplace=True)

        self.data["surgical_lesion"].replace({
            "yes":1,
            "no":0}
        ,inplace=True)

        self.data["surgery"].replace({
            "yes":1,
            "no":0}
        ,inplace=True)

        self.data["age"].replace({
            "adult":1,
            "young":0}
        ,inplace=True)
    
        self.y_vector=pd.get_dummies(self.data["outcome"])
        
        x_vector=self.Transformer.fit_transform(self.data)
        self.x_vector=pd.DataFrame(scipy.sparse.lil_matrix(x_vector).toarray())

    def save_data(self):
        self.x_vector.to_csv(self.config.transformerDataDestinationPath+"/x_train.csv")
        self.y_vector.to_csv(self.config.transformerDataDestinationPath+"/y_train.csv")    


if __name__=="__main__":


    try:

        logger.info("Data Transformation Initialized")
    
        obj=DataTransformation()
        logger.info("Loading Data")
        obj.load_data()
        logger.info("Data Extraction Successful")
        logger.info("Removing Unnecessary Features")
        obj.drop_columns()
        logger.info("Data Frame Built")
        logger.info("Building Transformer")

        obj.buildTransformer()
        logger.info("Transformer Built Successful")
        logger.info("Saving Transformer Successful")
        logger.info("Transforming Data")

        obj.transform_data()
        logger.info("Transformation Successful")
        obj.save_data()
        logger.info("Saved Data")
        logger.info("Data Transformation Terminated")

    except Exception as e:
        logger.critical(str(e))




    