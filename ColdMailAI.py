import os

# Disable CrewAI telemetry to avoid signal handler errors in Streamlit
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import re
import smtplib
from email.mime.text import MIMEText

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool

load_dotenv()

GEMINI_MODEL = "gemini/gemini-2.5-flash"
MAX_EMAIL_WORDS = 150
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

st.set_page_config(
    page_title="Cold Email Generator",
    page_icon="📧",
    layout="wide"
)


def is_valid_url(url: str) -> bool:
    pattern = re.compile(
        r'^(https?://)?'
        r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'
        r'(/.*)?$'
    )
    return bool(pattern.match(url.strip()))


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def get_llm() -> LLM:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file before running."
        )
    return LLM(model=GEMINI_MODEL, api_key=api_key)


def build_agents(llm: LLM, your_service: str):
    scrape_tool = ScrapeWebsiteTool()

    researcher = Agent(
        role="Business Intelligence Analyst",
        goal="Analyze the target company website and identify their core business and potential weaknesses.",
        backstory=(
            "You are an expert at analyzing businesses just by looking at their landing page. "
            "You look for what they do and where they might be struggling."
        ),
        tools=[scrape_tool],
        verbose=True,
        allow_delegation=False,
        memory=True,
        llm=llm
    )

    strategist = Agent(
        role="Service Strategist",
        goal="Match the target company needs with the service being offered.",
        backstory=(
            "You are an expert at identifying business needs and matching them with solutions. "
            "Your goal is to read the analysis of a prospect and determine how the following "
            f"service can help them:\n\nSERVICE BEING OFFERED:\n{your_service}\n\n"
            "You must explain why this service is a good fit for the target company based on "
            "their website analysis."
        ),
        verbose=True,
        memory=True,
        llm=llm
    )

    writer = Agent(
        role="Senior Sales Copywriter",
        goal="Write a personalized cold email that sounds human and professional.",
        backstory=(
            "You write emails that get replies. You never sound robotic. "
            "You mention specific details found by the Researcher to prove we actually "
            "looked at their site."
        ),
        verbose=True,
        memory=True,
        llm=llm
    )

    return researcher, strategist, writer


def build_tasks(researcher, strategist, writer, target_url, your_name, your_company, your_service):
    task_analyze = Task(
        description=(
            f"Scrape the website {target_url}. Summarize what the company does and identify "
            "1 key area where they could improve (e.g., design, traffic, automation)."
        ),
        expected_output="A brief summary of the company and their potential pain points.",
        agent=researcher
    )

    task_strategize = Task(
        description=(
            "Based on the analysis, pick ONE service from our Agency Knowledge Base that "
            "solves their problem. Explain the match."
        ),
        expected_output="The selected service and the reasoning for the match.",
        agent=strategist
    )

    task_write = Task(
        description=f"""Draft a personalized cold email from {your_name} at {your_company} to the target company.

Key requirements:
- Mention specific details found from their website to prove you researched them
- Explain how {your_service} can help solve their specific pain points
- Keep it under {MAX_EMAIL_WORDS} words
- Make it sound human and professional, not robotic
- Include a clear call-to-action""",
        expected_output="A professional cold email ready to send.",
        agent=writer
    )

    return [task_analyze, task_strategize, task_write]


def send_email(sender_email, sender_password, recipient_email, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


def generate_cold_email(your_name, your_company, your_service, target_url):
    llm = get_llm()
    researcher, strategist, writer = build_agents(llm, your_service)
    tasks = build_tasks(researcher, strategist, writer, target_url, your_name, your_company, your_service)

    sales_crew = Crew(
        agents=[researcher, strategist, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return sales_crew.kickoff()


def main():
    st.title("Cold Email AI")
    st.caption("Generate a personalized, research-backed cold email in one click.")

    st.sidebar.header("Your Information")
    your_name = st.sidebar.text_input("Your Name", placeholder="Enter your name")
    your_company = st.sidebar.text_input("Your Company", placeholder="Enter your company name")
    your_service = st.sidebar.text_area(
        "Your Service/Offering",
        placeholder="Describe what service or product you offer..."
    )

    st.header("Target Company Information")
    target_url = st.text_input("Enter Target Company Website URL", placeholder="https://example.com")

    if "last_email" not in st.session_state:
        st.session_state.last_email = None

    if st.button("Generate Cold Email", type="primary"):
        errors = []
        if not your_name:
            errors.append("Please enter your name in the sidebar.")
        if not your_company:
            errors.append("Please enter your company name in the sidebar.")
        if not your_service:
            errors.append("Please describe your service in the sidebar.")
        if not target_url:
            errors.append("Please enter a target company URL.")
        elif not is_valid_url(target_url):
            errors.append("That doesn't look like a valid URL. Try something like example.com")

        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        clean_url = normalize_url(target_url)

        with st.spinner("Researching the company and writing your email..."):
            try:
                result = generate_cold_email(your_name, your_company, your_service, clean_url)
                st.session_state.last_email = str(result)
            except ValueError as ve:
                st.error(str(ve))
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong while generating the email: {e}")
                st.stop()

    if st.session_state.last_email:
        st.subheader("Generated Cold Email")
        st.write(st.session_state.last_email)
        st.download_button(
            "Download as .txt",
            data=st.session_state.last_email,
            file_name="cold_email.txt",
            mime="text/plain"
        )

        st.divider()
        st.subheader("Send This Email")
        st.caption(
            "Sending uses your email account via SMTP. Set SENDER_EMAIL and "
            "SENDER_APP_PASSWORD in your .env file (use an app password, not your "
            "regular password)."
        )

        recipient_email = st.text_input("Recipient Email", placeholder="contact@targetcompany.com")
        email_subject = st.text_input("Subject", value=f"Quick idea for {your_company}" if your_company else "")

        if st.button("Send Email"):
            sender_email = os.getenv("SENDER_EMAIL")
            sender_password = os.getenv("SENDER_APP_PASSWORD")

            if not sender_email or not sender_password:
                st.error(
                    "Missing SENDER_EMAIL or SENDER_APP_PASSWORD in your .env file. "
                    "Add them before sending."
                )
            elif not recipient_email or not is_valid_url("http://" + recipient_email.split("@")[-1]):
                st.error("Please enter a valid recipient email address.")
            else:
                try:
                    with st.spinner("Sending email..."):
                        send_email(
                            sender_email,
                            sender_password,
                            recipient_email,
                            email_subject or "Quick idea for your team",
                            st.session_state.last_email
                        )
                    st.success(f"Email sent to {recipient_email}")
                except smtplib.SMTPAuthenticationError:
                    st.error("Login failed. Check your SENDER_EMAIL and app password.")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")


if __name__ == "__main__":
    main()
