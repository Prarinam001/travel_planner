import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agent import (
    budget_agent,
    flight_agent,
    hotel_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
    human_approval_agent,
    final_response_agent,
)
from configs import DATABASE_URL
from state import TravelState

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]