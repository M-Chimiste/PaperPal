from .prompting import prompt

INTERESTS_SCHEMA = """
{
    "related": "boolean",
    "rationale": "string",
    "score": "int from 1 to 10"
}
"""
NEWSLETTER_SCHEMA = """
{
    "draft": "string",
}
"""

SUMMARY_SCHEMA = """
{
    "questions": "string",
    "content": "string"
}
"""

SCRIPT_SCHEMA = """
[
    {
        "persona": "string - either Sarah, Alex, or Mike",
        "text": "string - the dialogue spoken by the character",
        "speaking_description": "string - a description of how the dialogue is delivered"
    },
    ...
    {
        "persona": "string - either Sarah, Alex, or Mike",
        "text": "string - the dialogue spoken by the character",
        "speaking_description": "string - a description of how the dialogue is delivered"
    }
]
"""

RESEARCH_INTERESTS_SYSTEM_PROMPT = f"""You are a research assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{INTERESTS_SCHEMA}\n</schema>"""

NEWSLETTER_SYSTEM_PROMPT = f"""You are an expert scientific author that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{NEWSLETTER_SCHEMA}\n</schema>"""

SYSTEM_CONTENT_EXTRACTION_SUMMARY = f"""You are an expert AI which helps extract and summarize content from text content. \
You only respond in JSON format. Here's the json schema you must adhere to:\n<schema>\n{SUMMARY_SCHEMA}\n</schema>."""

SUMMARY_SYSTEM_PROMPT = """You are an expert summarizing AI. You respond in JSON with the following structure: <schema>{"summary": <summary_text>}</schema>."""
