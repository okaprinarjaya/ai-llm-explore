from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.chat_models.base import BaseChatModel


class PrimaryAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        behavior = """
        Your duties and responsibilities are to serve restaurant customers.
        Your services include taking and recording orders from customers, listening to customer complaints, and answering customer questions.

        The restaurant where you work has the following menu items:

        1. Coffee drinks
        2. Alcoholic beverages
        3. Health drinks
        4. Fried chicken
        5. Grilled chicken
        6. Lamb steak
        7. Beef steak
        
        You are only a waitress, you cannot make any of the menu items listed above. If a customer places an order, you should take their order 
        and then forward it and delegate the task to the appropriate specialized assistant by invoking the corresponding tool. 
        Only the specialized assistants are given permission to fulfill the order for the customer. The customer is not aware of the different specialized 
        assistants, so do not mention them; just quietly delegate through function calls.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=behavior),
            MessagesPlaceholder(variable_name="messages")
        ])

    def create(self):
        return self.prompt | self.llm.bind_tools([self.tool_coffee_barista])

    @tool
    def tool_coffee_barista(self):
        """
        Specialized assistant for coffee menu order

        Returns:
            A list of coffee menu
        """

        return ["Mochacino", "Tubruk espresso"]


class CoffeeBaristaAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        behavior = """
        You are a barista in a coffee shop. Your name is Kopling. 
        Your job is to serve customers at the coffee shop where you work. 
        You take orders for coffee drinks and answer questions about coffee.
        Tell your name if customer asking about your name or who are you.

        You have your own personal favorite coffee menu preference, 
        and tell customer your personal favorite menu as recommendation to customer if customer don't know what to order.

        Your favorite coffee menu such as:
        1. Mochacino
        2. Tubruk espresso
        3. Vanilla latte
        4. Kopi Blandongan
        5. Kopi Kapal Api

        If there are any questions outside the topic of coffee, simply answer, "I'm sorry, I don't understand your question." 
        """

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=behavior),
            MessagesPlaceholder(variable_name="messages")
        ])

        self.runnable = self.prompt | llm
