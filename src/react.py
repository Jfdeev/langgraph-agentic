from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: int) -> int:
    """
    param num: The number to be cubed.
    return: The cubed value of the input number.
    """
    return num ** 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOllama(model="llama3.2:latest", temperature=0).bind_tools(tools)