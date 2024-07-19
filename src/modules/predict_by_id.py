from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sys
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel
from datetime import date
from typing import Optional
import json
import random
import  pickle
from keras.models import load_model
class Predict_by_id:
    def __init__(self,id):
        self.id = id

    @staticmethod
    def preprocess_data(df):
        # Преобразование строковых дат в формат datetime
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['finish_date'] = pd.to_datetime(df['finish_date'])

        # Извлечение признаков из дат
        df['start_month'] = df['start_date'].dt.month
        df['start_day_of_week'] = df['start_date'].dt.dayofweek
        df['finish_month'] = df['finish_date'].dt.month
        df['finish_day_of_week'] = df['finish_date'].dt.dayofweek

        # Создание относительного признака
        df['duration_days'] = (df['finish_date'] - df['start_date']).dt.days

        # Создание цикличных признаков
        def create_cyclical_features(df, column, period):
            df[column + '_sin'] = np.sin(2 * np.pi * df[column] / period)
            df[column + '_cos'] = np.cos(2 * np.pi * df[column] / period)

        create_cyclical_features(df, 'start_month', 12)
        create_cyclical_features(df, 'start_day_of_week', 7)
        create_cyclical_features(df, 'finish_month', 12)
        create_cyclical_features(df, 'finish_day_of_week', 7)

        # Удаление избыточных столбцов
        df.drop(columns=['start_date', 'finish_date'], inplace=True)

        return df

    @staticmethod
    def id_to_lstm(df,time_steps, id):
        scaler = pickle.load(open("/fastapi_app/src/data/models/cat/scaler.pkl", 'rb'))
        df = Predict_by_id.preprocess_data(df)
        sub_array = df[df['id'] == id].iloc[-time_steps:][
            ['shipper_st_code', 'consignee_st_code', 'cargo_code', 'cargo_weight', 'start_month', 'start_day_of_week',
             'finish_month',
             'finish_day_of_week', 'duration_days', 'start_month_sin',
             'start_month_cos', 'start_day_of_week_sin', 'start_day_of_week_cos',
             'finish_month_sin', 'finish_month_cos', 'finish_day_of_week_sin',
             'finish_day_of_week_cos']]
        sub_array = sub_array.to_numpy()
        if len(sub_array) < time_steps:
            sub_array = pad_sequences(sub_array.T, maxlen=time_steps, dtype='float32', padding='pre', value=0).T
        X2 = np.array(sub_array)
        X2 = scaler.transform(X2)
        return X2

    @staticmethod
    def id_to_name(dict1,dict2):
        return {value: dict1[key] for key, value in zip(dict1.keys(), dict2.values())}
    def get_result_by_id(self,full_df):
        dfw = Predict_by_id.id_to_lstm(full_df,5,self.id)
        model = load_model("/fastapi_app/src/data/models/lstm/model1905.h5")

        rw_output = model.predict(dfw.reshape(1, 5, 17))[0]
        services = pd.read_csv('/fastapi_app/src/data/service_ids.csv')
        services.loc[len(services.index)] = rw_output

        with open('/fastapi_app/src/data/code_recognized.json', 'r') as file:
            dict2 = json.load(file)
        status_option = "Заказывалось ранее"
        nums = random.randint(3, 10)
        newjs = {}
        ind = 1
        for i in services.columns:
            try:
                newjs[dict2[i]] = {"probability": float(services[i].iloc[0])}
            except:
                pass
            ind += 1
        sorted_dict = dict(sorted(newjs.items(), key=lambda item: item[1]['probability'], reverse=True))
        for ind, i in enumerate(sorted_dict):
            if ind <= nums:
                sorted_dict[i]['status'] = status_option
        return sorted_dict