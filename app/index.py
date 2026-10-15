from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.travel_advisor import GREETING, Itinerary, TravelAdvisor


load_dotenv()

app = FastAPI(
    title="AI Travel Advisor API",
    description="Plan personalised trips with an AI travel advisor.",
    version="1.0.0",
)

advisor = TravelAdvisor()


# ---------- What goes in and out (Pydantic) ----------

class Message(BaseModel):
    """One message of the conversation."""

    role: Literal["user", "assistant"]   # security: clients can't send "system" messages
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    """A new message plus the conversation so far."""

    message: str = Field(min_length=1, max_length=1500)
    history: list[Message] = Field(default_factory=list, max_length=40)


class ChatResponse(BaseModel):
    """The advisor's answer plus the updated conversation."""

    answer: str
    history: list[Message]


class ItineraryRequest(BaseModel):
    """The conversation that contains the trip plan."""

    history: list[Message] = Field(min_length=1, max_length=40)


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP service is running."""
    return {"status": "ok"}


@app.get("/greeting")
def greeting() -> dict[str, str]:
    """The first message the advisor shows."""
    return {"greeting": GREETING}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send one message. The client keeps the history and sends it back every time."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be blank.")

    history = [m.model_dump() for m in request.history]

    try:
        answer, history = advisor.chat(history, message)
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error

    return ChatResponse(answer=answer, history=history)


@app.post("/itinerary", response_model=Itinerary)
def itinerary(request: ItineraryRequest) -> Itinerary:
    """Turn the conversation's trip plan into a structured itinerary."""
    history = [m.model_dump() for m in request.history]

    try:
        return advisor.make_itinerary(history)
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error
