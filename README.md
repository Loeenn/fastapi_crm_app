# RZD_services_sorter

## Run

To run you need to have `columns.csv`, `df_final.csv`, `service_ids.csv` files in `./data/`

## Docker build

<strong> Warning! </strong> Sometimes you need to restart the container with `docker-compose up --build`, in main cases its 2-3 times

```bash
docker-compose up --build
```

By default app running on `http://localhost:8000`

## Swagger documentation

You can see documentation on `/docs` route or download on `/openapi.json` by default in `http://localhost:8000/docs`
