from fastapi import FastAPI
import process

app = FastAPI()

@app.get("/")
def root():
    process.main()
    return{"Hello": "World"}