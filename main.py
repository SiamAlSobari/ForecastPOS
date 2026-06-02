from fastapi import FastAPI
from app.api.controllers.stock_controller import stock_controller
from app.api.controllers.busy_hour_controller import busy_hour_controller
from app.api.controllers.insights_controller import insights_controller
from app.api.middlewares.auth import AuthMiddleware
app = FastAPI(title="ML Kasir API - Decision Support System")

# app.add_middleware(AuthMiddleware)
@app.get("/")
def home():
    return {"message": "API Kasir ML Aktif!", "port": 8080}


app.include_router(stock_controller, prefix="/api/predict")
app.include_router(busy_hour_controller, prefix="/api/predict")
app.include_router(insights_controller, prefix="/api/insights")

# Jika ingin menjalankan langsung via 'python main.py'
if __name__ == "__main__":
    import os
    import uvicorn

    workers = int(os.getenv("UVICORN_WORKERS", os.cpu_count() or 4))
    uvicorn.run("main:app", host="0.0.0.0", port=8080, workers=workers)