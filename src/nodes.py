from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSTEM_MENSSAGE="""You are helpful assistant that can use tools to answer user questions."""


def run_agent_resoning(state: MessagesState) -> MessagesState:
    """
        Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MENSSAGE}], *state.messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)