from fastapi import FastAPI
from app.api.controller.predict_controller import predict_controller

app = FastAPI(title="ML Kasir API - Decision Support System")


@app.get("/")
def home():
    return {"message": "API Kasir ML Aktif!", "port": 8080}


app.include_router(predict_controller, prefix="/api/predict")

# Jika ingin menjalankan langsung via 'python main.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)