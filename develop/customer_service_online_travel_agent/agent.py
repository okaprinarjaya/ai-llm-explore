# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage

from graph_cs_ota import State

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig) -> State:
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}

            result = self.runnable.invoke(state)

            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content 
                or isinstance(result.content, list) 
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}
    
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
