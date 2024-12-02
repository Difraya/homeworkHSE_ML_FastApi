import io
import pickle
from venv import logger

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

from starlette.responses import StreamingResponse

app = FastAPI()

with open("cars_model2.pickle", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
medians = model_data["medians"]

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def preprocessing(df: pd.DataFrame):
    try:
        logger.info("Начало предобработки данных.")
        logger.debug(f"Исходные данные:\n{df}")
        # Удаление цены
        if "selling_price" in df.columns:
            df = df.drop(columns=["selling_price"])
            logger.debug("Удалён столбец 'selling_price'.")
        # Обработка числовых данных
        df['mileage'] = df['mileage'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
        df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
        df['max_power'] = df['max_power'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
        df['max_power'] = df['max_power'].apply(lambda x: None if x == 0 else x)
        df['seats'] = df['seats'].astype(float)
        if 'torque' in df.columns:
            df.drop(columns='torque', inplace=True)
        # Заполнение пропусков
        df.fillna(medians, inplace=True)
        # Приведение к числовому типу
        cols = ['mileage', 'engine', 'max_power', 'seats']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df['engine'] = df['engine'].astype(int)
        df['seats'] = df['seats'].astype(int)
        df = df.select_dtypes(exclude='object')
        logger.debug(f"Данные после обработки:\n{df}")
        df_res = scaler.transform(df)
        logger.info("Предобработка завершена.")
        return df_res
    except Exception as e:
        logger.error(f"Ошибка при предобработке данных: {e}")
        raise ValueError(f"Ошибка при предобработке данных: {e}")

@app.get("/")
async def root():
    return {
        "Name": "Car price prediction",
        "description": "This is a cars price prediction model.",
    }


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    try:
        data = pd.DataFrame([item.dict()])
        logger.debug(f"Созданный DataFrame:\n{data}")
        formated_data = preprocessing(data)
        logger.debug(f"Предобработанные данные:\n{formated_data}")
        predict = model.predict(formated_data)
        logger.info(f"Предсказание: {predict[0]}")
        return float(predict[0])
    except ValueError as ve:
        logger.error(f"Ошибка в предобработке данных: {ve}")
        return {"error": f"Ошибка в предобработке данных: {ve}"}
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return {"error": f"Неожиданная ошибка: {e}"}

@app.post("/predict_csv")
async def predict_items(file: UploadFile = File(...)):
    text = await file.read()
    data_2 = pd.read_csv(io.BytesIO(text))
    formated_data = preprocessing(data_2)
    predict = model.predict(formated_data)
    data_2['predicted price'] = predict
    # сохраняем в csv
    stream = io.StringIO()
    data_2.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=prediction_price.csv"
    return response


