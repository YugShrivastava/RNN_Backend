from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TweetText(BaseModel):
    tweet: str

@app.post("/get-prediction")
async def get_prediction(tweet: TweetText):
    prediction = predict(tweet.tweet)
    return { "prediction": prediction }