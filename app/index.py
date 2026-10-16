import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.itinerary_pdf import itinerary_pdf
from app.travel_advisor import GREETING, Day, Itinerary, TravelAdvisor


load_dotenv()

app = FastAPI(
    title="AI Travel Advisor API",
    description="Plan personalised trips with an AI travel advisor.",
    version="1.0.0",
)

advisor = TravelAdvisor()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


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


class DayChangeRequest(BaseModel):
    """A finished itinerary, the day to change and what should change."""

    itinerary: Itinerary
    day_index: int = Field(ge=0, le=30)
    change: str = Field(min_length=1, max_length=300)


class ItineraryRequest(BaseModel):
    """The conversation that contains the trip plan."""

    history: list[Message] = Field(min_length=1, max_length=40)


# ---------- Endpoints ----------

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the web interface."""
    return FileResponse(STATIC_DIR / "index.html")


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


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Same as /chat, but the answer arrives piece by piece (one JSON object per line)."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be blank.")

    history = [m.model_dump() for m in request.history]

    def events():
        try:
            for event in advisor.chat_stream(history, message):
                yield json.dumps(event) + "\n"
        except Exception:
            # the HTTP status (200) is already sent, so the error travels as an event
            yield json.dumps({"type": "error", "message": "The AI provider could not finish the answer. Try again."}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/itinerary/day", response_model=Day)
def change_day(request: DayChangeRequest) -> Day:
    """Rewrite one day of the itinerary; the rest of the trip stays the same."""
    if request.day_index >= len(request.itinerary.days):
        raise HTTPException(status_code=422, detail="That day is not part of this itinerary.")
    change = request.change.strip()
    if not change:
        raise HTTPException(status_code=422, detail="Say what should change.")

    try:
        return advisor.rewrite_day(request.itinerary, request.day_index, change)
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error


@app.post("/itinerary", response_model=Itinerary)
def itinerary(request: ItineraryRequest) -> Itinerary:
    """Turn the conversation's trip plan into a structured itinerary."""
    history = [m.model_dump() for m in request.history]

    try:
        result = advisor.make_itinerary(history)
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error

    if not result.days:
        # the planner had too little to work with (for example, the advisor was still asking questions)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The advisor still needs a few details. Answer its last question, then create the itinerary.",
        )
    return result


@app.post("/itinerary/pdf", response_class=Response)
def itinerary_as_pdf(itinerary: Itinerary) -> Response:
    """Turn a finished itinerary into a designed PDF (no AI call, so it is free and fast)."""
    pdf = itinerary_pdf(itinerary)
    city = itinerary.destination.split(",")[0].strip() or "itinerary"
    safe_name = "".join(ch for ch in city if ch.isalnum() or ch in " -").strip() or "itinerary"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Meridian - {safe_name}.pdf"'},
    )
