import datetime as dt
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, Request
from pymongo import MongoClient
from pydantic.main import BaseModel
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

collection_name = "covid_scores"


def mongo_connector(mongo_path, db):
    client = MongoClient(mongo_path)
    db = client[db]
    return db


con = mongo_connector(
    'mongodb://mongo_user:mongo_password@mongo:27017', 'covid')

app = FastAPI(title="MongoDB Connector")


class PostScores(BaseModel):
    time: str
    model: str
    acc: float
    kappa: float
    f1mac: float
    f1mic: float


@app.get("/", status_code=200, summary="Returns 200 for healthcheck.", tags=["root"])
def index():
    return {"Still": "Alive!"}


@app.post("/addscore", summary="Adds validation metrics to MongoDB", response_description="Returns 200.", tags=["DB"])
def score_to_mongo(score: PostScores):
    result = con[collection_name].insert_one(score.dict())
    return {"insertion": result.acknowledged}
