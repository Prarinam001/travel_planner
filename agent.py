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


def hotel_agent(state: TravelState):
    query = f"Best hotels and areas to stay for: {state['user_query']}"

    print("\n================= HOTEL AGENT INPUT =================")
    print(query)
    print("=====================================================\n")

    result = asyncio.run(tavily_search(query))

    print("\n================== HOTEL SEARCH RESULT ==================")
    print(result)
    print("=========================================================\n")

    return {
        "hotel_results": str(result),
        "messages": [AIMessage(content="Hotel agent completed.")],
    }


def weather_agent(state: TravelState):
    constraints = state["trip_constraints"]
    city = constraints["destinations"]

    print("\n=============== WEATHER AGENT INPUT ====================")
    print("city: ", city)
    print("=======================================================\n")
    weather_data = asyncio.run(current_weather(city))
    forecast_data = asyncio.run(forecast(city))

    print("\n================== CURRENT WEATHER ==================")
    print(weather_data)
    print("=====================================================\n")

    print("\n================ FORECAST DATA ====================")
    print(forecast_data)
    print("====================================================\n")

    result = f"""
        Current weather:
        {weather_data}

        Current forecast:
        {forecast_data}
    """
    print("\n================== WEATHER AGENT OUTPUT ==================")
    print(result)
    print("==========================================================\n")

    return {
        "weather_results": result,
        "messages": [AIMessage(content="weather agent completed")]
    }


def budget_agent(state: TravelState):
    print("\n=============== BUDGET INPUT =====================")
    print("Trip Constraints")
    print(state.get("trip_constraints"))
    print("\nFlight Results")
    print(state.get("flight_results"))
    print("\nHotel Results")
    print(state.get("hotel_results"))
    print("\nWeather Results")
    print(state.get("weather_results"))
    print("====================================================\n")

    prompt = f"""
        Analyze weather this trip plan is realistic for the user's budget.

        user request:
        {state['user_query']}

        Constraints:
        {state.get("trip_constraints", {})}

        Flight Results:
        {state.get("flight_results", "")}

        Hotel Results:
        {state.get("hotel_results", "")}

        Weather Results:
        {state.get("weather_results", "")}

        Return a concise budget assesment with:
        1. estimate cost categories
        2. risk areas
        3. money saving tips
        4. weather the plan seems feasible
        """

    result = _llm_text(
        "You are a budget planning specialist.",
        prompt,
    )

    print("\n================== BUDGET AGENT OUTPUT ==================")
    print(result)
    print("==========================================================\n")

    return {
        "budget_results": result,
        "messages": [AIMessage(content="Budget agent completed")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: TravelState):
    print("\n=============== ITINERARY INPUT =====================")
    print("Trip Constraints")
    print(state.get("trip_constraints"))
    print("\nFlight Results")
    print(state.get("flight_results"))
    print("\nHotel Results")
    print(state.get("hotel_results"))
    print("\nWeather Results")
    print(state.get("weather_results"))
    print("\nBudget Results")
    print(state.get("budget_results"))
    print("====================================================\n")

    prompt = f"""
        Create a clear draft travel itinerary for the user.

        user request:
        {state['user_query']}

        Constraints:
        {state.get("trip_constraints", {})}

        Flight Results:
        {state.get("flight_results", "")}

        Hotel Results:
        {state.get("hotel_results", "")}

        Weather Results:
        {state.get("weather_results", "")}

        Budget Results:
        {state.get("budget_results", "")}

        Make the output structured, practical, and ready for human review.
        """

    result = _llm_text(
        "You are an expert travel itinerary planning specialist.",
        prompt,
    )

    print("\n================== ITINERARY AGENT OUTPUT ==================")
    print(result)
    print("==========================================================\n")

    approval_request = f"""
        Please review this draft travel plan.

        {result}

        Reply with approval or feedback.
        """

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def human_approval_agent(state: TravelState):
    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            }
        }
    )

    approved = feedback.get("approved", False)
    human_feedback = feedback.get("feedback", "")

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [HumanMessage(content="Human approval received.")]
    }


def final_response_agent(state: TravelState):
    print("\n=============== FINAL RESPONSE INPUT =====================")
    print("Approved: ", state.get("approved"))
    print("Human Feedback: ", state.get("human_feedback"))
    print("============================================================\n")

    if state.get("approved"):
        prompt = f"""
            The human approved the draft itinerary.

            Producr the final polished travel plan.

            Draft Itinerary:
            {state.get("itinerary", "")}

            Budget Notes:
            {state.get("budget_results", "")}
        """
    else:
        prompt = f""""
        The human didn't approved the draft itineary.

        Original user request:
        {state.get("user_query", "")}

        Draft Itinerary:
        {state.get("itinerary", "")}

        Human Feedback:
        {state.get("human_feedback", "")}

        Budget Notes:
        {state.get("budget_results", "")}

        """

    result = _llm_text(
        "You produce final user-ready travel plans.",
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
