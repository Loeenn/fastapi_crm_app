import uvicorn
from fastapi import FastAPI, Body
from modules.predict_by_data import Predict_by_data
from modules.predict_by_id import Predict_by_id
import pandas as pd
from typing import Dict, Annotated, Union
from fastapi.middleware.cors import CORSMiddleware
from sqladmin import Admin, ModelView
from typing import List
import json
from fastapi_users import FastAPIUsers

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from auth.auth import auth_backend
from auth.manager import get_user_manager
from auth.schemas import UserRead, UserCreate
from auth.database import get_async_session, get_user_db, get_order_db, Order, User, Role, async_session_maker, \
    Search_history, Order, AsyncSessionLocal, engine
from models.models import *
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import HTTPException, status, Depends
from typing import Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
import datetime
from datetime import date
from sqlalchemy import func
from sqlalchemy.future import select
from sqlalchemy import distinct
import time
import logging

logging.basicConfig(filename='logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def http_exception_handler(request: Request, exc: HTTPException, data=None):
    if exc.status_code == 401:
        # Кастомное сообщение только для ошибок 401
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": "Пользователь не авторизован.", "request": data},
        )
    else:
        # Для всех остальных HTTP ошибок используем стандартный обработчик
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )


app = FastAPI()
admin = Admin(app, engine)


class UserAdmin(ModelView, model=User):
    column_list = [User.id, User.hashed_id, User.email, User.username, User.registered_at, User.role_id]


class OrderAdmin(ModelView, model=Order):
    column_list = [Order.order_id, Order.client_hid, Order.start_date, Order.finish_date, Order.shipper_st_code,
                   Order.consignee_st_code, Order.cargo_code, Order.cargo_weight, Order.services, Order.status]


admin.add_view(UserAdmin)
admin.add_view(OrderAdmin)

app.add_exception_handler(HTTPException, http_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://localhost:80', 'http://localhost'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# codes = pd.read_excel("data/codes_recognized.xlsx")
# if codes['code_number'].duplicated().any():
#     print("Обнаружены дубликаты в колонке 'id'.")
#     # Удаление дубликатов, оставляя первое вхождение
#     codes = codes.drop_duplicates(subset='code_number', keep='first')
# json_str = codes.set_index('code_number')['code_name'].to_json(force_ascii=False)
# with open('data/code_recognized.json', 'w') as f:
#     f.write(json_str)


# Определение PUT запроса@app.put("/predict")
full_df = pd.read_csv("data/df_small.csv")
with open("data/ordered.json", 'r') as f:
    ordered = json.load(f)
fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

current_user = fastapi_users.current_user()


async def get_bd_df():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(Order.start_date, Order.finish_date, Order.shipper_st_code,
                       Order.consignee_st_code, Order.cargo_code, Order.cargo_weight, Order.client_hid).select_from(Order))
            ordrs = result.fetchall()
            df = pd.DataFrame(ordrs, columns=[
                "start_date",
                "finish_date",
                "shipper_st_code",
                "consignee_st_code",
                "cargo_code",
                "cargo_weight",
                "id"
            ])
            return df


@app.on_event("startup")
async def init_db():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            roles_count = await session.execute(select(func.count()).select_from(Role))
            cnt = roles_count.scalar()

            orders_count = await session.execute(select(func.count()).select_from(Order))
            ordscnt = orders_count.scalar()

            if cnt == 0:
                print("Записи отсутствуют")
                user_role = Role(id=1, name="user")
                admin_role = Role(id=2, name="admin")
                session.add_all([user_role, admin_role])

            if ordscnt == 0:
                for index, row in full_df.iterrows():
                    order = Order(
                        client_hid=row['id'],
                        start_date=pd.to_datetime(row['start_date']).date(),
                        finish_date=pd.to_datetime(row['finish_date']).date(),
                        shipper_st_code=row['shipper_st_code'],
                        consignee_st_code=row['consignee_st_code'],
                        cargo_code=row['cargo_code'],
                        cargo_weight=row['cargo_weight'],
                        status="Выполнен",
                        services=ordered[index]
                    )
                    session.add(order)
            else:
                print("Записи присутствуют")

        await session.commit()


@app.post("/predict_authorized")
async def predict_authorized(task: Predict_by_data, user: User = Depends(current_user),
                             session: AsyncSession = Depends(get_async_session)) -> (
        Annotated[Dict[str, Union[ServiceStatus,ServiceStatusUnathorized]], Body(
            examples={
                "example1": {
                    "summary": "Пример вывода",
                    "value": {
                        "Отправление грузобагажа": {"probability": 0.5, "status": "Заказано"},
                        "Провозная плата": {"probability": 0.3, "status": "Заказано"}
                    }
                }
            }
        )]):
    start_time = time.time()
    task.authorized = True
    data_dict = task.dict()
    # Оборачиваем каждое скалярное значение в список
    keys = list(data_dict.keys())[:-2]  # убираем модел тайп и ауторайзед
    df_dict = {}
    for key in keys:
        df_dict[key] = [data_dict[key]]
    df_dict['id'] = str(user.hashed_id)
    df = pd.DataFrame(df_dict)
    bd_df = await get_bd_df()
    #result = task.get_result(df, bd_df)
    result = task.get_custom_result(df, bd_df)
    entity = Search_history(
        client_hid=str(user.hashed_id),
        search_time=date.today(),
        doc_state="поиск",
        start_date=pd.to_datetime(task.start_date).date(),
        finish_date=pd.to_datetime(task.finish_date).date(),
        shipper_st_code=task.shipper_st_code,
        consignee_st_code=task.consignee_st_code,
        cargo_code=task.cargo_code,
        cargo_weight=task.cargo_weight
    )
    session.add(entity)
    await session.commit()
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Время выполнения функции predict_authorized: {execution_time} секунд")
    return result


@app.post("/predict_unauthorized")
async def predict_unauthorized(task: Predict_by_data) -> (Annotated[Dict[str, ServiceStatusUnathorized], Body(
    examples={
        "example1": {
            "summary": "Пример вывода",
            "value": {
                "Отправление грузобагажа": {"probability": 0.5},
                "Провозная плата": {"probability": 0.3}
            }
        }
    }
)]):
    start_time = time.time()
    task.authorized = False
    data_dict = task.dict()
    # Оборачиваем каждое скалярное значение в список
    keys = list(data_dict.keys())[:-2]  # убираем модел тайп и ауторайзед
    df_dict = {}
    for key in keys:
        df_dict[key] = [data_dict[key]]
    df = pd.DataFrame(df_dict)
    result = task.get_result_cat(df)
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Время выполнения функции predict_unauthorized: {execution_time} секунд")
    return result


# для чела
@app.post("/predict_by_id")
async def predict_by_id(user: User = Depends(current_user), session: AsyncSession = Depends(get_async_session)) -> (
        Annotated[Dict[str, Union[str, Union[ServiceStatus,ServiceStatusUnathorized]]], Body(
            examples={
                "user_id": "defrr",
                "example1": {
                    "summary": "Пример вывода",
                    "value": {
                        "Отправление грузобагажа": {"probability": 0.5, "status": "Заказано"},
                        "Провозная плата": {"probability": 0.3, "status": "Заказано"}
                    }
                }
            }
        )]):
    if not user or not hasattr(user, 'hashed_id'):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    task = Predict_by_id(str(user.hashed_id))
    bd_df = await get_bd_df()
    return task.get_result_by_id(bd_df)


# для манагера
@app.post("/predict_by_id_foreign")
async def predict_by_id_foreign(id: str, user: User = Depends(current_user),
                                session: AsyncSession = Depends(get_async_session)) -> (
        Annotated[Dict[str, Union[str, Union[ServiceStatus,ServiceStatusUnathorized]]], Body(
            examples={
                "user_id": "defrr",
                "example1": {
                    "summary": "Пример вывода",
                    "value": {
                        "Отправление грузобагажа": {"probability": 0.5, "status": "Заказано"},
                        "Провозная плата": {"probability": 0.3, "status": "Заказано"}
                    }
                }
            }
        )]):
    if user.role_id != 2:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    task = Predict_by_id(str(id))
    bd_df = await get_bd_df()
    return task.get_result_by_id(bd_df)


@app.get("/get_foreign_history_orders")
async def get_foreign_history_orders(id: str, user: User = Depends(current_user),
                                     session: AsyncSession = Depends(get_async_session)):
    if user.role_id != 2:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    else:
        result = await session.execute(select(Order).filter(Order.client_hid == str(id)))
        history_data = result.scalars().all()
        if len(history_data) == 0:
            return JSONResponse(content=[])
        else:
            return JSONResponse(content=[
                {"client_hid": record.client_hid,
                 "start_date": record.start_date.isoformat(),
                 "finish_date": record.finish_date.isoformat(), "shipper_st_code": record.shipper_st_code,
                 "consignee_st_code": record.consignee_st_code, "cargo_code": record.cargo_code,
                 "cargo_weight": record.cargo_weight, "services": record.services,"status": record.status} for record in history_data])


# -> Annotated[Dict[str,str], Body(examples={"frfe":"Компания_1","fdf43":"Компания_2"})]
@app.get("/get_ids")
async def get_ids(session: AsyncSession = Depends(get_async_session)):
    async with session.begin():
        result = await session.execute(select(distinct(Order.client_hid)))
        unique_ids = result.scalars().all()
        comp_names = [f"Компания_{i}" for i in range(1, len(unique_ids) + 1)]
        return dict(zip(unique_ids, comp_names))


@app.get("/get_shipper_st_codes")
async def get_shipper_st_codes(session: AsyncSession = Depends(get_async_session)):
    async with session.begin():
        result = await session.execute(select(distinct(Order.shipper_st_code)))
        st_codes = result.scalars().all()
        return st_codes


@app.get("/get_consignee_st_codes")
async def get_consignee_st_codes(session: AsyncSession = Depends(get_async_session)):
    async with session.begin():
        result = await session.execute(select(distinct(Order.consignee_st_code)))
        st_codes = result.scalars().all()
        return st_codes


@app.get("/get_cargo_codes")
async def get_cargo_codes(session: AsyncSession = Depends(get_async_session)):
    async with session.begin():
        result = await session.execute(select(distinct(Order.cargo_code)))
        cargo_codes = result.scalars().all()
        return cargo_codes


@app.get("/get_my_orders")
async def read_user_orders(user: User = Depends(current_user), session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(Order).filter(Order.client_hid == str(user.hashed_id)))
    history_data = result.scalars().all()
    if len(history_data) == 0:
        return JSONResponse(content=[])
    else:
        return JSONResponse(content=[
            {"client_hid": record.client_hid,
            "start_date": record.start_date.isoformat(),
             "finish_date": record.finish_date.isoformat(), "shipper_st_code": record.shipper_st_code,
             "consignee_st_code": record.consignee_st_code, "cargo_code": record.cargo_code,
             "cargo_weight": record.cargo_weight, "services": record.services,"status": record.status} for record in history_data])


@app.get("/account/me")
async def get_info(user: User = Depends(current_user), session: AsyncSession = Depends(get_async_session)):
    return JSONResponse(content={"username": f"{user.username}", "hashed_id": user.hashed_id, "role_id": user.role_id,
                                 "email": user.email, "registered_at": user.registered_at.isoformat()})


@app.get("/search_history")
async def search_history(id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_async_session)):
    if user.role_id != 2:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    result = await session.execute(select(Search_history).filter(Search_history.client_hid == id))
    history_data = result.scalars().all()
    if len(history_data) == 0:
        raise HTTPException(status_code=404, detail="Tickets not found")
    return JSONResponse(content=[
        {"search_id": record.search_id, "client_hid": record.client_hid, "search_time": record.search_time.isoformat(),
         "doc_state": record.doc_state, "start_date": record.start_date.isoformat(),
         "finish_date": record.finish_date.isoformat(), "shipper_st_code": record.shipper_st_code,
         "consignee_st_code": record.consignee_st_code, "cargo_code": record.cargo_code,
         "cargo_weight": record.cargo_weight, "services": record.services} for record in history_data])


@app.post("/create_order")
async def create_order(order_params: OrderParams, user: User = Depends(current_user),
                       session: AsyncSession = Depends(get_async_session)) -> (Annotated[Dict[str, str], Body(
    examples={
        "addedServices": {
            "orderParameters": {
                "start_date": "2022-01-18",
                "finish_date": "2022-01-29",
                "shipper_st_code": 123213,
                "consignee_st_code": 123123123,
                "cargo_code": 12312312,
                "cargo_weight": 123123123
            },
            "services": [
                {
                    "name": "За подачу и уборку вагонов на МОП",
                    "count": 4
                },
                {
                    "name": "Под/уборка вагонов на СВХ и ЗТК при погр/выгр.ваг.",
                    "count": 2
                },
                {
                    "name": "За непредъяв.грузов в соответ.с назначением(вагон)",
                    "count": 1
                }
            ]
        }
    })]):
    new_order = Order(
        client_hid=str(user.hashed_id),
        start_date=pd.to_datetime(order_params.addedServices["orderParameters"]["start_date"]).date(),
        finish_date=pd.to_datetime(order_params.addedServices["orderParameters"]["finish_date"]).date(),
        shipper_st_code=order_params.addedServices["orderParameters"]["shipper_st_code"],
        consignee_st_code=order_params.addedServices["orderParameters"]["consignee_st_code"],
        cargo_code=order_params.addedServices["orderParameters"]["cargo_code"],
        cargo_weight=float(order_params.addedServices["orderParameters"]["cargo_weight"]),
        services=order_params.addedServices["services"],
        status="Создан"
    )
    session.add(new_order)
    await session.commit()
    await session.refresh(new_order)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Заказ добавлен"}
    )


@app.get("/get_uncompleted_orders")
async def get_uncompleted_orders(user: User = Depends(current_user),
                                 session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(Order).filter(Order.status == "Создан"))
    orders_users = result.scalars().all()
    orders_json = [i.__dict__ for i in orders_users]
    return orders_json


# Запуск сервера с помощью Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
