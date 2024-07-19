FROM python:3.10

WORKDIR /fastapi_app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
COPY ./app.sh ./app.sh

RUN chmod a+x ./*.sh

# RUN alembic upgrade head
# ENTRYPOINT cd ./src && gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000

# ENTRYPOINT /bin/bash -c source ./app.sh
# ENTRYPOINT ./app.sh

#WORKDIR src

#CMD gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000