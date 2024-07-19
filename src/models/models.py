from datetime import datetime

from sqlalchemy import MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean, Float
from pydantic import BaseModel, Field
from typing import Dict, Any
from typing import List
from datetime import date

metadata = MetaData()

role = Table(
    "role",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("permissions", JSON, nullable=True),
)

user = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("hashed_id", String, nullable=True, default="", unique=True),
    Column("email", String, nullable=False),
    Column("username", String, nullable=False),
    Column("registered_at", TIMESTAMP, default=datetime.utcnow),
    Column("role_id", Integer, ForeignKey(role.c.id), default=1),
    Column("hashed_password", String, nullable=False),
    Column("is_active", Boolean, default=True, nullable=False),
    Column("is_superuser", Boolean, default=False, nullable=False),
    Column("is_verified", Boolean, default=False, nullable=False),
)

orders = Table(
    "orders",
    metadata,
    Column("order_id", Integer, primary_key=True, autoincrement=True),
    Column("client_hid", String),
    Column("start_date", TIMESTAMP, nullable=False),
    Column("finish_date", TIMESTAMP, nullable=True),
    Column("shipper_st_code", Integer, nullable=False),
    Column("consignee_st_code", Integer, nullable=False),
    Column("cargo_code", Integer, nullable=False),
    Column("cargo_weight", Float, nullable=False),
    Column("services", JSON, nullable=True),
    Column("status", String, nullable=True, default="Создан")
)
search_history = Table(
    'search_history',
    metadata,
    Column("search_id", Integer, primary_key=True, autoincrement=True),
    Column("client_hid", String, ForeignKey(user.c.hashed_id)),
    Column("search_time", TIMESTAMP, nullable=True),
    Column("doc_state", String, nullable=False),
    Column("start_date", TIMESTAMP, nullable=False),
    Column("finish_date", TIMESTAMP, nullable=True),
    Column("shipper_st_code", Integer, nullable=False),
    Column("consignee_st_code", Integer, nullable=False),
    Column("cargo_code", Integer, nullable=False),
    Column("cargo_weight", Float, nullable=False),
    Column("services", JSON, nullable=True)
)


class ServiceStatus(BaseModel):
    probability: float = Field(..., example=0.5)
    status: str = Field(..., example="Заказано")


class ServiceStatusUnathorized(BaseModel):
    probability: float = Field(..., example=0.5)


class AddedService(BaseModel):
    name: str
    count: float


# Определяем модель для параметров заказа
class OrderParameters(BaseModel):
    start_date: str
    finish_date: str
    shipper_st_code: int
    consignee_st_code: int
    cargo_code: int
    cargo_weight: float


# Определяем модель для всего JSON-объекта
class OrderParams(BaseModel):
    addedServices: dict
