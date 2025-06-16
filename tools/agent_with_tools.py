import os 
from langchain_tavily import TavilySearch
from langchain_community.utilities import SearxSearchWrapper

from langchain_core.tools import Tool
from langchain import hub
import langgraph
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools import PythonREPLTool
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from langgraph.graph import END
from typing import Union
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.tools import tool
import random
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import AgentExecutor
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from tools.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import init_chat_model

from tools.text_inspector_tool import TextInspectorTool
from tools.visual_qa import VisualQATool

load_dotenv()

##############
# TOOLS
##############

# SEARCH TOOL
# web_search_tool = TavilySearch(max_results=3, search_depth = 'advanced')
# search_searx = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

python_repl_tool = PythonREPLTool()

###############
# SUPERVISOR AGENT
###############
    
def create_supervisor_agent(llm, tools = []):


    # Create the ReAct agent
    supervisor_agent = create_react_agent(
        model=llm,
        tools=tools,
        name="supervisor_agent"
    )

    def execute_step(state):
        # print('EXECUTION STEP')
        question = state["question"]
        previous_steps_with_answers = "\n".join(
            f"{i}. INSTRUCTION: {state['plan'][i]} ANSWER: {state['intermediate_responses'][i]}" for i in range(len(state['intermediate_responses']))
        )
        current_step = state["plan"][state["current_step"]]

        prompt = f"""
            You are a helpful agent that should solve the instruct given to you by the user. You have multiple tools available to you. \n\n """
            # If you want to look something up on the Internet, you should use: GoogleSerperAPIWrapper. It is good to browse on the internet, it gives you a list of relevant websites and a short snippet of those.\n
            # If you think that there is a website that you found that might contain relevant information for you, you should use these tools to navigate a website and find relevant information: visit_page, page_up, page_down, find_on_page_ctrl_f, find_next.\n
            # If you want to download a file from the internet for later inspection, then you should use: download_file.\n
            # If you want to inspect a file as markdown text, then you should use: inspect_file_as_text.\n
            # If you want to analyse an image and have questions answered about it, then you should use: visualizer.\n
            # If you want to do math operations then you should use: add, multiply, divide or substract. \n
            # If you want to execute some python code, then you should use: PythonREPLTool.\n
            # \n
            # """
        human_message = f"""
            The instruction is only a single step of a plan that solves a bigger problem. You should not return the final answer.\n
            Your task is only to answer to the instruction given to you. You shouldn't add other text than the answer.\n
            The final goal of the plan is to solve the following question: \n
            {question}\n"""
        
        if len(previous_steps_with_answers) > 0:
            human_message += f"""
            You have already executed some steps of the plan. \n
            You can consider the previous steps and their answers when executing the current step.\n
            The previous instructions of the plan that have already been solved and their answers :\n
            {previous_steps_with_answers}\n
            """

        human_message += f"""
            You need to solve this instruction: {current_step}\n
            """

        input_message = {"messages": [
            SystemMessage(content=prompt),
            HumanMessage(content=human_message)
        ]}

        output = supervisor_agent.invoke(
            input_message, 
        )

        # Access intermediate steps
        intermediate_steps = output.get("intermediate_steps", [])
        for step in intermediate_steps:
            print(step)
        print('Agent executed step:', output)
        result = output["messages"][-1].content
        new_response = state["intermediate_responses"] + [result]
        new_step = state["current_step"] + 1
        return {"intermediate_responses": new_response, "current_step": new_step}

    return execute_step

##############
# STATE
##############
class PlanExecute(TypedDict):
    question: str
    plan: List[str]
    intermediate_responses: List[str]
    response: str
    current_step: int
    error_count: int
    validation: None
    agent_finished: bool


###############
# PLANNER
###############
def create_planner_agent(llm):

    def plan_step(state: PlanExecute):
        # planner_prompt = """
        # You are a planner, and your goal is to develop a plan to solve the users requests.
        # Break down the complex question into small executable steps for an AI Agent. 
        # For each step, provide a brief description of what needs to be done, there should be enough information to execute the step.
        # The steps should be VERY simple and easy to execute. You should consider creating many short steps. Consider required tools (web search, calculator, file processing, code execution)
        # You should NOT return a numbered list. Return only the steps, do not add any other text.
        # """
        planner_prompt = f"""
        You are a planning assistant tasked with decomposing complex user requests into a sequence of simple, executable steps for an AI agent.\n
        - Clarity and Simplicity: Break down the user’s request into the smallest possible actions. Each step should be straightforward and unambiguous.\n
        - Tool Specification: For each step, specify any required tools or resources (e.g., web search, calculator, file processing, code execution).\n
        - When calling a tool to process a file, don't create a step that just opens it. Create a step and ask a question about it. 
	    - Sequential Order: Ensure that the steps are ordered logically, with each step building upon the previous ones if necessary.\n
	    - Format: List each step on a new line without numbering or additional commentary.\n
        """

        messages = [
            SystemMessage(content=planner_prompt),
            HumanMessage(content=state["question"])
        ]
        plan = llm(messages)
        plan_list = [step.strip() for step in plan.content.split("\n") if step.strip()]
        # print('PLANNER LIST')
        # for i, step in enumerate(plan_list):
        #     print(f"{i+1}. {step}")
        # print(' ')
        return {"plan": plan_list}
    
    return plan_step

 

###############
# RE PLANNER
###############
def create_replanner_agent(llm):
    def replan_step(state: PlanExecute):

        previous_steps_with_answers = "\n".join(
            f"{i}. INSTRUCTION: {state['plan'][i]} ANSWER: {state['intermediate_responses'][i]}/n" for i in range(len(state['intermediate_responses']))
        )
        future_steps = "\n".join(
            f"{i}. INSTRUCTION: {state['plan'][i]}/n" for i in range(len(state['intermediate_responses']), len(state['plan']))
        )

        # print("REPLANNING inside agent")
        # print('PRevious steps with answer')
        # for i in range(len(state['intermediate_responses'])):
        #     print(f"{i+1}. INSTRUCTION: {state['plan'][i]} ANSWER: {state['intermediate_responses'][i]}")
        # print('future steps')
        # for i in range(len(state['intermediate_responses']), len(state['plan'])):
        #     print(f"{i+1}. INSTRUCTION: {state['plan'][i]}")
        # print(' ')
        
        question = state['question']
        
        replanner_prompt = f""" 
        You are a planner, and your objective is to develop a plan to solve the users question.
        Break down the complex question into small executable steps for an AI Agent. 
        For each step, provide a brief description of what needs to be done, there should be enough information to execute the step.
        The steps should be simple and easy to follow. Consider required tools (web search, calculator, file processing, code execution)
        A plan has already been created, and some steps have been executed. You should only update the future steps of the plan. 
        You cannot change the past steps of the plan.
        You should NOT return a numbered list. Return only the steps, do not add any other text.
        If you think that with the information gathered during the previous steps, one can answer the users question, then answer "FINISH".
        """

        human_message = f"""QUESTION: {question}.\n\n

        The PAST executed steps with their answers are:
        {previous_steps_with_answers} \n\n

        The planned FUTURE steps that are in the plan are, do you think they are still relevant? If not, you should update them:
        {future_steps}\n\n
        Remember, don't return a numbered list, just return the steps. 
        """

        messages = [
            SystemMessage(content=replanner_prompt),
            HumanMessage(content=human_message)
        ]
        # print('REPLAN STEPS input', messages)
        output = llm(messages)
        # print('REPLAN STEPS output', output)
        if "FINISH" in output.content:
            # print('AGENT FINISHED')
            # print('output content', output.content)
            return {"agent_finished": True}
        else:
            # print('CHANGE NEEDED')
            # print('output content', output.content)
            plan_list = [step.strip() for step in output.content.split("\n") if step.strip()]
            updated_plan = state['plan'][:len(state['intermediate_responses'])] + plan_list
            # print('UPDATED PLAN')
            # for i in range(len(updated_plan)):
                # print(f"{i+1}. {updated_plan[i]}")
            # print(' ')
            return {"plan": updated_plan}
    return replan_step

################
# ANSWER AGENT
################
def create_answer_agent(llm):
    def answer_step(state: PlanExecute):
        # print("ANSWERING")
        # print('PRevious steps with answer')
        # for i in range(len(state['intermediate_responses'])):
        #     print(f"{i+1}. INSTRUCTION: {state['plan'][i]} ANSWER: {state['intermediate_responses'][i]}")

        question = state['question']
        previous_steps_with_answers = "\n".join(
            f"{i}. PAST INSTRUCTION: {state['plan'][i]}. ANSWER: {state['intermediate_responses'][i]}" for i in range(len(state['intermediate_responses']))
        )
        answer_prompt = f""" You are an answer agent. Your role is to provide the final answer to the question based on the previous tasks and their answers. \n
                        You will find the answer to the question in the previous steps and their answers. You should certainly have a look at the last steps.\n

                        Finish your answer with the following template: 
                        FINAL ANSWER: [YOUR FINAL ANSWER]. 
                        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
                        Your answer should only start with "FINAL ANSWER: ", then follows with the answer. You response should only contain the answer, do not add any other text.
                        """  

        human_message = f"""
        The initial question is: {question}.\n\n

        The executed steps with their answers are:
        {previous_steps_with_answers} \n\n
        Remember, you should only return the final answer to the question, do not add any other text. Use the template: FINAL ANSWER: [YOUR FINAL ANSWER].
        """

        messages = [
            SystemMessage(content=answer_prompt),
            HumanMessage(content=human_message)
        ]

        # print('ANSWER STEPS input', messages)
        output = llm(messages)
        # print('ANSWER STEPS output', output)
        
        return {"response": output.content}
    return answer_step

###############
# GRAPH
###############
def create_plan_and_execute_agent(
    llm_name_planner: str,
    llm_name_replanner: str,
    llm_name_executor: str,
    llm_name_answer: str,
):

    llm_planner = ChatOpenAI(
        temperature=0,
        model=llm_name_planner,
        streaming=True,
        verbose=True,
    )
    llm_replanner = ChatOpenAI(
        temperature=0,
        model=llm_name_replanner,
        streaming=True,
        verbose=True,
    )
    llm_answer = ChatOpenAI(
        temperature=0,
        model=llm_name_answer,
        streaming=True,
        verbose=True,
    )
    search_google_serperAPIwrapper = GoogleSerperAPIWrapper()
    google_search_tool =  Tool(
            name="GoogleSerperAPIWrapper",
            func=search_google_serperAPIwrapper.results,
            description="Good tool if you want to browse the internet. It returns a list of websites and a snippet of their content.",
        )
    
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    BROWSER_CONFIG = {
        "viewport_size": 1024 * 30,
        "downloads_folder": "downloads_folder",
        "request_kwargs": {
            "headers": {"User-Agent": user_agent},
            "timeout": 300,
        },
        "serpapi_key": os.getenv("SERPAPI_API_KEY"),
    }
    os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    tools = [
        google_search_tool,
        python_repl_tool,
        VisitTool(browser=browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(
            model=init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0), 
            text_limit=10000),
        VisualQATool(model=init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0)),
    ]
    llm_executor = ChatOpenAI(
        temperature=0,
        model=llm_name_executor,
        streaming=True,
        verbose=True,
    )

    planner = create_planner_agent(llm_planner)
    replanner = create_replanner_agent(llm_replanner)
    agent_executor = create_supervisor_agent(llm = llm_executor, tools=tools)
    answer_agent = create_answer_agent(llm_answer)


    def should_end(state: PlanExecute):
        # print('SHOULD END? ????')
        # print('state', state)
        if state["agent_finished"]:
            # print('START END')
            return 'answer'
        else:
            # print('NOT END')
            return "agent"
    

    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", planner)

    # Add the execution step
    workflow.add_node("agent", agent_executor)

    # Add a replan node
    workflow.add_node("replan", replanner)

    # Add a final answer node
    workflow.add_node("answer", answer_agent)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", "answer"],
    )
    workflow.add_edge("answer", END)


    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    agent = workflow.compile()
    return agent

# test
if __name__ == "__main__":
    print('Start agent')
    # Build the graph
    graph = create_plan_and_execute_agent(
            llm_name_planner="gpt-4.1-mini",
            llm_name_executor="gpt-4.1-mini",
            llm_name_replanner="gpt-4.1-mini",
            llm_name_answer="gpt-4.1-mini")
    print('Graph builded')

    # Run the graph
    question= "What was the complete title of the book in which two James Beard Award winners recommended the restaurant where Ali Khan enjoyed a New Mexican staple in his cost-conscious TV show that started in 2015? Write the numbers in plain text if there are some in the title."
    question = "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
    # question = "what is the hometown of the mens 1999 Australia open winner?"
    # question = "what is 12034 * 1485033 + 5.2**3 ?"
    print('question', question)

    config = {"recursion_limit": 50}
    inputs = {"input": question}
    initial_state = {
        "question": question,
        "plan": [],
        "response": "",
        "current_step": 0,
        "error_count": 0,
        "validation": None,
        "agent_finished": False
    }
    initial_state = PlanExecute(
        question=question,
        plan=[],
        intermediate_responses=[],
        response="",
        current_step=0,
        error_count=0,
        validation=None,
        agent_finished=False
    )

    print('start invoke')
    workflow =  graph.invoke(initial_state, config=config)

    print('workflow', workflow)
    print(' ')
    print('RESPONSE')
    print(workflow["response"])
    print(' ')

    print('PLAN')
    print(workflow["plan"])
    print(' ')

    print('STEPS AND ANSWERS')
    for i in range(len(workflow["plan"])):
        step = workflow["plan"][i]
        response = workflow["intermediate_responses"][i]
        print(f"{i+1}. INSTRUCTION: {step}. ANSWER: {response}")
    print(' ')

    print('QUESTION')
    print(workflow["question"])

    print(' ')
    print('current step')
    print(workflow["current_step"])
