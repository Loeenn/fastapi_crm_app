from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel
from datetime import date
from typing import Optional
import json
import random
import pickle
import catboost
from keras.models import load_model
from catboost import CatBoostClassifier


class Predict_by_data(BaseModel):
    start_date: date
    finish_date: date
    shipper_st_code: int
    consignee_st_code: int
    cargo_code: int
    cargo_weight: float
    id: Optional[str] = None
    models_type: Optional[str] = "/fastapi_app/src/data/models/lstm"
    authorized: Optional[bool] = False

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
    def row_to_lstm(df, time_steps, row):
        time_steps -= 1
        df = Predict_by_data.preprocess_data(df)
        scaler = pickle.load(open("/fastapi_app/src/data/models/cat/scaler.pkl", 'rb'))
        id = row['id'].item()
        sub_array = df[df['id'] == id].iloc[-time_steps:][
            ['shipper_st_code', 'consignee_st_code', 'cargo_code', 'cargo_weight', 'start_month', 'start_day_of_week',
             'finish_month',
             'finish_day_of_week', 'duration_days', 'start_month_sin',
             'start_month_cos', 'start_day_of_week_sin', 'start_day_of_week_cos',
             'finish_month_sin', 'finish_month_cos', 'finish_day_of_week_sin',
             'finish_day_of_week_cos']]
        row = row[
            ['shipper_st_code', 'consignee_st_code', 'cargo_code', 'cargo_weight', 'start_month', 'start_day_of_week',
             'finish_month',
             'finish_day_of_week', 'duration_days', 'start_month_sin',
             'start_month_cos', 'start_day_of_week_sin', 'start_day_of_week_cos',
             'finish_month_sin', 'finish_month_cos', 'finish_day_of_week_sin',
             'finish_day_of_week_cos']].to_numpy()
        sub_array = sub_array.to_numpy()
        if len(sub_array) < time_steps:
            sub_array = pad_sequences(sub_array.T, maxlen=time_steps, dtype='float32', padding='pre', value=0).T
        sub_array = np.append(sub_array, row)
        X2 = np.array(sub_array.reshape(5, 17))
        X2 = scaler.transform(X2)
        return X2

    @staticmethod
    def id_to_name(dict1, dict2):
        return {value: dict1[key] for key, value in zip(dict1.keys(), dict2.values())}

    def get_result(self, df, full_df):
        print("LSTM")
        df1 = Predict_by_data.preprocess_data(df)
        dfw = Predict_by_data.row_to_lstm(full_df, 5, df1)
        model = load_model("/fastapi_app/src/data/models/lstm/model1905.h5")

        rw_output = model.predict(dfw.reshape(1, 5, 17))[0]
        services = pd.read_csv('/fastapi_app/src/data/service_ids.csv')
        services.loc[len(services.index)] = rw_output

        with open('/fastapi_app/src/data/code_recognized.json', 'r') as file:
            dict2 = json.load(file)
        if not self.authorized:
            newjs = {}
            for i in services.columns:
                try:
                    newjs[dict2[i]] = {"probability": float(services[i].iloc[0])}
                except:
                    pass
            sorted_dict = dict(sorted(newjs.items(), key=lambda item: item[1]['probability'], reverse=True))
            return sorted_dict
        else:
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
            for ind,i in enumerate(sorted_dict):
                if ind<=nums:
                    sorted_dict[i]['status'] = status_option
            return sorted_dict
    def get_custom_result(self, df, full_df):
        print("Custom LSTM")
        df1 = Predict_by_data.preprocess_data(df)
        dfw = Predict_by_data.row_to_lstm(full_df, 5, df1)
        model = load_model("/fastapi_app/src/data/models/lstm/model1905.h5")

        rw_output = model.predict(dfw.reshape(1, 5, 17))[0]
        services = pd.read_csv('/fastapi_app/src/data/service_ids.csv')
        services.loc[len(services.index)] = rw_output

        with open('/fastapi_app/src/data/code_recognized.json', 'r') as file:
            dict2 = json.load(file)
        if not self.authorized:
            newjs = {}
            for i in services.columns:
                try:
                    newjs[dict2[i]] = {"probability": float(services[i].iloc[0])}
                except:
                    pass
            sorted_dict = dict(sorted(newjs.items(), key=lambda item: item[1]['probability'], reverse=True))
            return sorted_dict
        else:
            status_option = "Заказывалось ранее"
            nums = random.randint(3, 10)
            newjs = {}
            ind = 1
            for i in services.columns:
                try:
                    if "плата" in dict2[i] or "Плата" in dict2[i] or str(i) == str(1108717875) or "VipNet" in dict2[i] or "За" in dict2[i]:
                        newjs[dict2[i]] = {"probability": float(services[i].iloc[0])/100}
                    else:
                        newjs[dict2[i]] = {"probability": float(services[i].iloc[0])}
                except:
                    pass
                ind += 1
            sorted_dict = dict(sorted(newjs.items(), key=lambda item: item[1]['probability'], reverse=True))
            for ind,i in enumerate(sorted_dict):
                if ind<=nums:
                    sorted_dict[i]['status'] = status_option
            return sorted_dict
    @staticmethod
    def get_ids(df):
        return df['id'].unique().tolist()

    @staticmethod
    def get_company_names(df):
        idlists = df['id'].unique().tolist()
        comp_names = [f"Компания_{i}" for i in range(1, len(idlists) + 1)]
        return dict(zip(idlists, comp_names))

    @staticmethod
    def get_shipper_st_codes(df):
        return df['shipper_st_code'].unique().tolist()

    @staticmethod
    def get_consignee_st_codes(df):
        return df['consignee_st_code'].unique().tolist()

    @staticmethod
    def get_cargo_codes(df):
        return df['cargo_code'].unique().tolist()

    @staticmethod
    def row_to_cat(row):
        scaler = pickle.load(open("/fastapi_app/src/data/models/cat/scaler.pkl", 'rb'))
        row = Predict_by_data.preprocess_data(row)
        row = row.drop(columns=['id'])
        row = row[
            ['shipper_st_code', 'consignee_st_code', 'cargo_code', 'cargo_weight', 'start_month', 'start_day_of_week',
             'finish_month', 'finish_day_of_week', 'duration_days',
             'start_month_sin', 'start_month_cos', 'start_day_of_week_sin',
             'start_day_of_week_cos', 'finish_month_sin', 'finish_month_cos',
             'finish_day_of_week_sin', 'finish_day_of_week_cos']]
        row = scaler.transform(row)
        return row

    def get_result_cat(self, df):

        df = Predict_by_data.row_to_cat(df)
        model = CatBoostClassifier()
        model.load_model("/fastapi_app/src/data/models/cat/catboost_last.cbm")

        rw_output = model.predict_proba(df)[0]
        services = pd.read_csv('/fastapi_app/src/data/service_ids.csv')
        services.loc[len(services.index)] = rw_output

        with open('/fastapi_app/src/data/code_recognized.json', 'r') as file:
            dict2 = json.load(file)
        if not self.authorized:
            newjs = {}
            for i in services.columns:
                try:
                    newjs[dict2[i]] = {"probability": float(services[i].iloc[0])}
                except:
                    pass
            sorted_dict = dict(sorted(newjs.items(), key=lambda item: item[1]['probability'], reverse=True))
            return sorted_dict
        else:
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
            for ind,i in enumerate(sorted_dict):
                if ind<=nums:
                    sorted_dict[i]['status'] = status_option
            return sorted_dict
