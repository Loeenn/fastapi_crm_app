from datetime import datetime
from typing import AsyncGenerator

from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from sqlalchemy import Column, String, Boolean, Integer, TIMESTAMP, ForeignKey, Float, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker

# from src.Config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
DB_HOST = "db"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "postgres"
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

Base: DeclarativeMeta = declarative_base()


class Role(Base):
    __tablename__ = 'role'
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    name = Column("name", String, nullable=False)
    permissions = Column("permissions", JSON, nullable=True)


class User(SQLAlchemyBaseUserTable[int], Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, autoincrement=True)
    hashed_id = Column("hashed_id", String, nullable=True,unique=True)
    email = Column(String, nullable=False)
    username = Column(String, nullable=False)
    registered_at = Column(TIMESTAMP, default=datetime.utcnow)
    role_id = Column(Integer, ForeignKey("role.id"))
    hashed_password: str = Column(String(length=1024), nullable=False)
    is_active: bool = Column(Boolean, default=True, nullable=False)
    is_superuser: bool = Column(Boolean, default=False, nullable=False)
    is_verified: bool = Column(Boolean, default=False, nullable=False)


class Order(Base):
    __tablename__ = 'orders'
    order_id = Column("order_id", Integer, primary_key=True, autoincrement=True)
    client_hid = Column("client_hid", String)
    start_date = Column("start_date", TIMESTAMP, nullable=False)
    finish_date = Column("finish_date", TIMESTAMP, nullable=True)
    shipper_st_code = Column("shipper_st_code", Integer, nullable=False)
    consignee_st_code = Column("consignee_st_code", Integer, nullable=False)
    cargo_code = Column("cargo_code", Integer, nullable=False)
    cargo_weight = Column("cargo_weight", Float, nullable=False)
    services = Column("services", JSON, nullable=True)
    status = Column("status",String,nullable=True,default="Создан")


class Search_history(Base):
    __tablename__ = 'search_history'
    search_id = Column("search_id", Integer, primary_key=True, autoincrement=True)
    client_hid = Column("client_hid", String, ForeignKey("user.hashed_id"))
    search_time = Column("search_time", TIMESTAMP, nullable=True)
    doc_state = Column("doc_state", String, nullable=False)
    start_date = Column("start_date", TIMESTAMP, nullable=False)
    finish_date = Column("finish_date", TIMESTAMP, nullable=True)
    shipper_st_code = Column("shipper_st_code", Integer, nullable=False)
    consignee_st_code = Column("consignee_st_code", Integer, nullable=False)
    cargo_code = Column("cargo_code", Integer, nullable=False)
    cargo_weight = Column("cargo_weight", Float, nullable=False)
    services = Column("services", JSON, nullable=True)


engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


async def get_order_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, Order)

AsyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)