import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

agency_services = """
1. AI Automation: Best for companies with manual, repetitive tasks. We build agents to save time.
2. AI Agents: Best for companies needing smart assistants for support, sales, and daily tasks.
3. AI Problem Solving: Best for companies facing workflow inefficiencies. We provide practical AI solutions.
"""

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

scrape_tool = ScrapeWebsiteTool()

researcher = Agent(
    role='Business Intelligence Analyst',
    goal='Analyze the target company website and identify their core business and potential weaknesses.',
    backstory="You are an expert at analyzing businesses just by looking at their landing page. You look for what they do and where they might be struggling.",
    tools=[scrape_tool],
    verbose=True,
    allow_delegation=True,
    memory=True,
    llm=llm
)

strategist = Agent(
    role='Agency Strategist',
    goal='Match the target company needs with ONE of our agency services.',
    backstory=f"""You work for a top-tier digital agency.
Your goal is to read the analysis of a prospect and decide which of OUR services to pitch.

OUR SERVICES KNOWLEDGE BASE:
{agency_services}

You must pick the SINGLE best service for this specific client and explain why.""",
    verbose=True,
    memory=True,
    llm=llm
)

writer = Agent(
    role='Senior Sales Copywriter',
    goal='Write a personalized cold email that sounds human and professional.',
    backstory="""You write emails that get replies. You never sound robotic.
You mention specific details found by the Researcher to prove we actually looked at their site.""",
    verbose=True,
    llm=llm
)

target_url = "https://openai.com/"

task_analyze = Task(
    description=f"Scrape the website {target_url}. Summarize what the company does and identify 1 key area where they could improve (e.g., design, traffic, automation).",
    expected_output="A brief summary of the company and their potential pain points.",
    agent=researcher
)

task_strategize = Task(
    description="Based on the analysis, pick ONE service from our Agency Knowledge Base that solves their problem. Explain the match.",
    expected_output="The selected service and the reasoning for the match.",
    agent=strategist
)

task_write = Task(
    description="Draft a cold email to the CEO of the target company. Pitch the selected service. Keep it under 150 words.",
    expected_output="A professional cold email ready to send.",
    agent=writer
)

sales_crew = Crew(
    agents=[researcher, strategist, writer],
    tasks=[task_analyze, task_strategize, task_write],
    process=Process.sequential,
    verbose=True
)

result = sales_crew.kickoff()

print(result)
