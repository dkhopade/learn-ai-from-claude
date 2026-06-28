from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI week 1 — hello world"}
