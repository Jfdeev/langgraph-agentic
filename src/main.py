from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

from nodes import run_agent_resoning, tool_node

load_dotenv()


AGENT_REASON="agent_reasoning"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    """
    Determine whether to continue the agent loop or end it.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


graph = StateGraph(MessagesState)

graph.add_node(AGENT_REASON, run_agent_resoning)
graph.set_entry_point(AGENT_REASON)
graph.add_node(ACT, tool_node)

graph.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT
})

graph.add_edge(ACT, AGENT_REASON)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")




if __name__ == "__main__":
    res = app.invoke({"messages": [HumanMessage(content="What is the wheader in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)
