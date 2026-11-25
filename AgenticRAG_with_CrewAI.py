"""Agentic RAG with CrewAI (ตาม notebook AgenticRAG_with_CrewAI.ipynb ปรับให้ใช้ Claude + MiniLM แทน OpenAI)

5 agent เรียงต่อกัน: Rewrite -> Router -> Retriever -> Evaluator -> Answer
  - Router เลือกแหล่ง: "vector_store" (ค้นเว็บของเรา) หรือ "web_search" (DuckDuckGo)
  - Evaluator ตัดสินว่า context พอตอบไหม (yes/no) -> Answer ตอบ หรือบอกว่า "Unable to answer"

ต้อง: pip install crewai crewai-tools langchain-community ddgs
      (แนะนำลงใน venv แยก: crewai ดึง dependency เยอะ อาจชนกับ langchain-chroma/numpy ในเครื่อง)
ต้องมี ANTHROPIC_API_KEY ใน .env
ยังไม่ได้รันจริง: config `embedder` ของ WebsiteSearchTool เปลี่ยนตามเวอร์ชัน crewai ถ้า error ดูตรง TOOL_CONFIG
"""
from pathlib import Path
from typing import Any, Type

from crewai import LLM, Agent, Crew, Task
from crewai.tools import BaseTool
from crewai_tools import WebsiteSearchTool
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")  # อ่าน ANTHROPIC_API_KEY

MODEL = "anthropic/claude-haiku-5-5"  # รูปแบบ litellm: provider/model
SITE = "https://www.akmmusai.pro/"  # เว็บที่ใช้เป็น "vector store" (ของ notebook)

# WebsiteSearchTool = ดึงเว็บ -> chunk -> embed -> ค้น (RAG สำเร็จรูปในตัว tool)
# crewai-tools 1.x: key ชื่อ embedding_model (ไม่ใช่ embedder) ถ้าผิด key จะ fallback ไป OpenAI แล้วขอ OPENAI_API_KEY
# embedding รัน local ด้วย MiniLM; tool นี้ไม่ใช้ LLM (Claude ใช้ที่ agent ผ่าน llm ด้านล่าง)
TOOL_CONFIG = dict(
    embedding_model=dict(provider="sentence-transformer", config=dict(model_name="all-MiniLM-L6-v2")),
)


# ---------- Tools ----------
website_search_tool = WebsiteSearchTool(website=SITE, config=TOOL_CONFIG)


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGoSearch tool."""

    query: str = Field(..., description="The search query to use with DuckDuckGo.")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Searches the web using DuckDuckGo for the provided query."
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> Any:
        return DuckDuckGoSearchResults(output_format="list").run(query)


web_search_tool = DuckDuckGoSearchTool()

# ---------- Agents ----------
llm = LLM(model=MODEL)


def make_agent(role: str, goal: str, backstory: str, tools=None) -> Agent:
    return Agent(role=role, goal=goal, backstory=backstory, llm=llm, tools=tools or [],
                 allow_delegation=False, verbose=True)


rewrite_agent = make_agent(
    "Query Rewriting Specialist",
    "Analyze and improve search query for better results",
    "Expert in natural language processing and query optimization with years of experience in improving search effectiveness.",
)
router_agent = make_agent(
    "Traffic Router",
    "Determine the appropriate search destination based on query content",
    "Specialized in query analysis and routing with deep understanding of various data sources and their applicability.",
)
retriever_agent = make_agent(
    "Information Retriever",
    "Fetch relevant information from appropriate source",
    "Experienced in efficient information retrieval from multiple sources with expertise in both web and vector store searches.",
    tools=[website_search_tool, web_search_tool],
)
evaluator_agent = make_agent(
    "Content Evaluator",
    "Assess the relevance and completeness of retrieved information",
    "Expert in content analysis and quality assessment with strong analytical skills and attention to detail.",
)
answer_agent = make_agent(
    "Response Generator",
    "Generate clear and accurate answers",
    "Skilled in synthesizing information and crafting clear, concise responses while maintaining accuracy and relevance.",
)

# ---------- Tasks (context = ผลของ task ก่อนหน้าที่ส่งต่อให้) ----------
rewrite_task = Task(
    description="""Analyze the input query: {query}, and rewrite it if needed to improve search effectiveness.
    Keep the query as is if it's already well-formed. Consider adding relevant keywords
    and removing unnecessary words while maintaining the original intent.""",
    expected_output="""A well-formed query that maintains the original intent but is optimized
    for search. Return the original query if no rewriting is needed.""",
    agent=rewrite_agent,
)
router_task = Task(
    description="""Analyze the query to determine if it's related to Akmmus AI Labs.
    Return 'vector_store' if the query is about Akmmus AI Labs, otherwise return 'web_search'.
    Look for keywords related to Akmmus, AI Labs, or specific projects/products.""",
    expected_output="""A string containing either 'vector_store' or 'web_search' based on
    the query analysis. The output should be lowercase and match exactly one of these two options.""",
    agent=router_agent,
    context=[rewrite_task],
)
retriever_task = Task(
    description="""Use the appropriate search tool based on the router's output to fetch relevant information.
    For 'vector_store', search Website search tool. For 'web_search', use web search tool.""",
    expected_output="""A string containing the search results. The output should include relevant
    information that directly addresses the query, with sources when available.""",
    agent=retriever_agent,
    context=[rewrite_task, router_task],
)
evaluator_task = Task(
    description="""Analyze the retrieved information to determine if it's relevant to the query,
    and contains a sufficient answer. Consider factors like content relevance, completeness,
    and reliability of the information.""",
    expected_output="""A string containing 'yes' if the content is relevant and contains
    a sufficient answer, or 'no' if the content is irrelevant or insufficient.""",
    agent=evaluator_agent,
    context=[rewrite_task, retriever_task],
)
answer_task = Task(
    description="""Generate a comprehensive answer to the query based on the retrieved information.
    If the evaluator returned 'yes', synthesize the information into a clear and concise response.
    If 'no', return 'Unable to answer the given query'.""",
    expected_output="""A string containing either a well-formatted answer based on the retrieved
    information or 'Unable to answer the given query' if the information was deemed insufficient.""",
    agent=answer_agent,
    context=[rewrite_task, retriever_task, evaluator_task],
)

# ---------- Crew ----------
agentic_rag_crew = Crew(
    agents=[rewrite_agent, router_agent, retriever_agent, evaluator_agent, answer_agent],
    tasks=[rewrite_task, router_task, retriever_task, evaluator_task, answer_task],
    verbose=True,
)

if __name__ == "__main__":
    result = agentic_rag_crew.kickoff(inputs={"query": "Who is the founder of Akmmus AI Labs?"})
    print(result)
