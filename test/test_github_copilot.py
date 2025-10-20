import os
from dotenv import load_dotenv
load_dotenv()
from langchain_github_copilot import ChatGitHubCopilot

llm = ChatGitHubCopilot()
llm.invoke("Tell me about twenty one pilots").pretty_print()

# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# llm.invoke("Sing a ballad of GitHub and LangChain.").pretty_print()