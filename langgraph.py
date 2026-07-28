from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ==========================================================
# LLM
# ==========================================================

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3
)

MAX_RETRIES = 3

# ==========================================================
# STATE
# ==========================================================

class AgentState(TypedDict):
    user_query: str

    research_context: str

    research_summary: str

    draft_response: str

    feedback: Optional[str]

    approved: bool

    retry_count: int


# ==========================================================
# RESEARCH AGENT TOOLS
# ==========================================================

class ResearchTools:

    @staticmethod
    def rag_retrieval(query: str):

        """
        Replace with your existing RAG.
        """

        docs = retriever.invoke(query)

        return "\n".join(
            doc.page_content
            for doc in docs
        )

    @staticmethod
    def extract_key_points(context):

        prompt = f"""
        Extract key points from:

        {context}
        """

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content

    @staticmethod
    def validate_context(context):

        prompt = f"""
        Validate whether this context
        is useful for answering a query.

        Context:
        {context}

        Return improved context.
        """

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content


# ==========================================================
# CONTENT WRITER TOOLS
# ==========================================================

class WriterTools:

    @staticmethod
    def create_response(
        query,
        research_summary,
        feedback=None
    ):

        if feedback:

            prompt = f"""
            Query:
            {query}

            Research:
            {research_summary}

            Reviewer Feedback:
            {feedback}

            Rewrite response.
            """

        else:

            prompt = f"""
            Query:
            {query}

            Research:
            {research_summary}

            Generate answer.
            """

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content

    @staticmethod
    def summarize_response(response_text):

        prompt = f"""
        Improve clarity and structure.

        {response_text}
        """

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content

    @staticmethod
    def format_response(response_text):

        return f"""
### Final Draft

{response_text}
"""


# ==========================================================
# MITL TOOLS
# ==========================================================

class MITLTools:

    @staticmethod
    def quality_check(response_text):

        prompt = f"""
        Evaluate quality of response.

        Response:
        {response_text}

        Mention weaknesses.
        """

        result = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return result.content

    @staticmethod
    def policy_check(response_text):

        prompt = f"""
        Check policy compliance.

        Response:
        {response_text}
        """

        result = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return result.content

    @staticmethod
    def human_review(response_text):

        print("\nGenerated Response:")
        print(response_text)

        print("\n1. Approve")
        print("2. Reject")

        choice = input("\nSelect: ")

        if choice == "1":

            return {
                "approved": True,
                "feedback": ""
            }

        feedback = input(
            "Reason for rejection: "
        )

        return {
            "approved": False,
            "feedback": feedback
        }


# ==========================================================
# RESEARCH AGENT
# ==========================================================

def research_agent(state):

    query = state["user_query"]

    context = ResearchTools.rag_retrieval(
        query
    )

    validated_context = (
        ResearchTools.validate_context(
            context
        )
    )

    summary = (
        ResearchTools.extract_key_points(
            validated_context
        )
    )

    print("Research Agent Completed")

    return {
        "research_context":
        validated_context,

        "research_summary":
        summary
    }


# ==========================================================
# CONTENT WRITER AGENT
# ==========================================================

def writer_agent(state):

    response = WriterTools.create_response(
        state["user_query"],
        state["research_summary"],
        state.get("feedback")
    )

    response = WriterTools.summarize_response(
        response
    )

    response = WriterTools.format_response(
        response
    )

    print("Writer Agent Completed")

    return {
        "draft_response": response
    }


# ==========================================================
# MITL AGENT
# ==========================================================

def mitl_agent(state):

    quality = MITLTools.quality_check(
        state["draft_response"]
    )

    print("\nQuality Report:")
    print(quality)

    compliance = MITLTools.policy_check(
        state["draft_response"]
    )

    print("\nPolicy Report:")
    print(compliance)

    review = MITLTools.human_review(
        state["draft_response"]
    )

    return {
        "approved":
        review["approved"],

        "feedback":
        review["feedback"],

        "retry_count":
        state["retry_count"] + (
            0 if review["approved"]
            else 1
        )
    }


# ==========================================================
# ROUTER
# ==========================================================

def approval_router(state):

    if state["approved"]:

        return "approved"

    if state["retry_count"] >= MAX_RETRIES:

        return "stop"

    return "rewrite"


# ==========================================================
# GRAPH
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node(
    "research_agent",
    research_agent
)

builder.add_node(
    "writer_agent",
    writer_agent
)

builder.add_node(
    "mitl_agent",
    mitl_agent
)

builder.add_edge(
    START,
    "research_agent"
)

builder.add_edge(
    "research_agent",
    "writer_agent"
)

builder.add_edge(
    "writer_agent",
    "mitl_agent"
)

builder.add_conditional_edges(
    "mitl_agent",
    approval_router,
    {
        "approved": END,
        "rewrite":
            "writer_agent",
        "stop":
            END
    }
)

graph = builder.compile()

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    result = graph.invoke(
        {
            "user_query":
            input("Enter Query: "),

            "research_context": "",

            "research_summary": "",

            "draft_response": "",

            "feedback": None,

            "approved": False,

            "retry_count": 0
        }
    )

    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)

    print(
        result["draft_response"]
    )
