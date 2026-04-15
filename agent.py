import asyncio
import os
from typing import Any

import dotenv
from github import Github
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

# --- GitHub & repo setup ---
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

repository = os.getenv("REPOSITORY")          # e.g. "aDonky/recipes-api"
pr_number = int(os.getenv("PR_NUMBER", "0"))  # PR number passed by GitHub Actions

repo = git.get_repo(repository) if git is not None and repository else None

# --- LLM setup ---
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


# --- GitHub tool functions ---

def get_pr_details(pr_number: int) -> dict:
    """Fetch details of a pull request given its number. Returns the author, title,
    body, diff_url, state, and commit SHAs of the PR."""
    if repo is None:
        return {"error": "GitHub repo not available."}

    pull_request = repo.get_pull(pr_number)

    commit_SHAs = []
    commits = pull_request.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_SHAs": commit_SHAs,
    }


def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its path.
    Returns the file content as a string."""
    if repo is None:
        return "GitHub repo not available."

    try:
        contents = repo.get_contents(file_path)
        return contents.decoded_content.decode("utf-8")
    except Exception as e:
        return f"Error fetching file '{file_path}': {e}"


def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Fetch details of a commit given its SHA. Returns a list of changed files
    with filename, status, additions, deletions, changes, and patch (diff)."""
    if repo is None:
        return [{"error": "GitHub repo not available."}]

    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post a review comment to a GitHub pull request given its number and the review comment body."""
    if repo is None:
        return "GitHub repo not available."

    try:
        pull_request = repo.get_pull(pr_number)
        pull_request.create_review(body=comment)
        return f"Review successfully posted to PR #{pr_number}."
    except Exception as e:
        return f"Error posting review to PR #{pr_number}: {e}"


# --- State management functions ---

async def add_context_to_state(ctx: Context, context: str) -> str:
    """Useful for saving the gathered context summary to the shared workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["gathered_contexts"] = context
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for saving the drafted PR review comment to the shared workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for saving the final reviewed PR comment to the shared workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."


# --- Convert GitHub functions to tools ---
pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

# --- ContextAgent ---
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository including PR details, changed files, commit info, and file contents.",
    tools=[pr_details_tool, file_contents_tool, commit_details_tool, add_context_to_state],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather \n: "
        "  - The details: author, title, body, diff_url, state, and head_sha; \n"
        "  - Changed files; \n"
        "  - Any requested for files; \n"
        "Once you gather the requested info, you MUST hand control back to the Commentor Agent. "
    ),
    can_handoff_to=["CommentorAgent"],
)

# --- CommentorAgent ---
commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n "
        "Ensure to do the following for a thorough review: \n"
        " - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. \n"
        " - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n"
        "    - What is good about the PR? \n"
        "    - Did the author follow ALL contribution rules? What is missing? \n"
        "    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n"
        "    - Are new endpoints documented? - use the diff to determine this. \n "
        "    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n"
        " - If you need any additional details, you must hand off to the ContextAgent. \n"
        " - You should directly address the author. So your comments should sound like: \n"
        " \"Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?\"\n"
        " - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. \n"
        "CRITICAL: You MUST NEVER output a final text response. Your workflow MUST always end by: \n"
        "  1. Calling add_comment_to_state with your complete draft review text. \n"
        "  2. IMMEDIATELY handing off to ReviewAndPostingAgent. \n"
        "  Do NOT stop or output any text directly — always use tools and handoffs."
    ),
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

# --- ReviewAndPostingAgent ---
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment generated by the CommentorAgent, requests rewrites if necessary, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt=(
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. \n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n"
        "   - The review must: \n"
        "   - Be a ~200-300 word review in markdown format. \n"
        "   - Specify what is good about the PR: \n"
        "   - Did the author follow ALL contribution rules? What is missing? \n"
        "   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n"
        "   - Are there notes on whether new endpoints were documented? \n"
        "   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
        " If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
        " When you are satisfied: \n"
        "   1. Call add_final_review_to_state with the final review text. \n"
        "   2. Call post_review_to_github with the PR number and the final review text. \n"
        "CRITICAL: You MUST always call post_review_to_github to post the review. Do NOT just output text."
    ),
    can_handoff_to=["CommentorAgent"],
)

# --- AgentWorkflow ---
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)


# --- Async main (no input — uses env vars from GitHub Actions) ---
async def main():
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    if git:
        git.close()
