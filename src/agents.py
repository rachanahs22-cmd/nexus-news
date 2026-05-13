import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, End
from langchain_ollama import ChatOllama