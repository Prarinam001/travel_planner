import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import interrupt

from configs import get_llm
from mcp_client import current_weather, forecast, list_airlines, list_airports, tavily_search
from state import TravelState


llm = get_llm()

def _llm_text(system: str, prompt: str)->str:
    response = llm.invoke(
        [
            SystemMessage(content = system),
            HumanMessage(content = prompt)
        ]
    )
    return response.content

def _json_from_llm(text: str) -> dict:
    print("\n=========== RAW LLM RESPONSE =============")
    print(text)
    print("============================================")

    start = text.index("{")
    end = text.rindex("}")+1
    json_text = text[start:end]

    print("\n========== EXTRACTED JSON ================")
    print(json_text)
    print("============================================")

    return json.loads(json_text)

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    prompt = f"""
        You are the supervisor of a real-world multi-agent travel planning system.
        Decide which specilaist agents are needed for this user request.

        Available Agents:
        - flight_agent: use when flights, airports, airlines, routes or airfare guidance are needed.
        - hotel_agent: use hotels, stays neighbourhoods, or accommodation are needed
        - weather_agent: use when weather, climate, seasons, packing, or forecast is useful
        - budget_agent: use when budget, affordability, cost or price constraints are mentioned
        - itinerary_agent: almost always needed to produce the travel plan

        Return only JSON with this Schema:

        {{
        "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
        "trip_constraints": {{
            "destination": "",
            "origin": "",
            "duration": "",
            "budget": "",
            "travel_style": "",
            "special_preferences": []
        }},
        "reasoning": ""
        }}

        User request:
        {query}
    """
    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        prompt,
    )
    print("\n===================== RAW LLM RESPONSE ====================")
    print(raw)
    print(type(raw))
    print("============================================================\n")

    parsed = _json_from_llm(raw)
    print("\n===================== PARSED LLM RESPONSE ====================")
    print(parsed)
    print(type(parsed))
    print("============================================================\n")

    selected = parsed["selected_agents"]
    return {
        "selected_agents": selected,
        "trip_constraints": parsed["trip_constraints"],
        "supervisor_reasoning": parsed["reasoning"],
        "messages":[AIMessage(content="Supevisor created the agent plan.")],
        "llm_calls":state.get("llm_calls", 0)+1
    }

def flight_agents(state: TravelState):
    query = state["user_query"]
    constraints = state["trip_constraints"]
    destination = constraints["destination"]

    print("\n================ FLIGHT AGENT INPUT====================")
    print("Query: ", query)
    print("Constraints: ", constraints)
    print("=========================================================")

    airports = asyncio.run(list_airports(destination, limit=10))
    airlines = asyncio.run(list_airlines("", limit=10))

    print("\n=============== AIRPORT MCP DATA ====================")
    print(airports)
    print("======================================================\n")

    print("\n================= AIRLINE MCP DATA ===================")
    print(airlines)
    print("=======================================================\n")

    prompt = f"""
        Create flight guidance for the trip.

        User request:
        {query}

        Trip Constraints:
        {constraints}

        Airport MCP data:
        {str(airports)[:3000]}

        Airlines MCP data:
        {str(airlines)[:3000]}

        Include likely departure/arrival airports, relevant airlines,
        estimated duration, fare range, peak season warning,
        and booking advice.
    """
    result = _llm_text(
        "You are a flight planning specialist.",
        prompt,
    )

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
