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

RESEARCH_INTERESTS_SYSTEM_PROMPT = f"""You are a research assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{INTERESTS_SCHEMA}\n</schema>"""
NEWSLETTER_SYSTEM_PROMPT = f"""You are an expert scientific author that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{NEWSLETTER_SCHEMA}\n</schema>"""


@prompt
def research_interests_prompt(research_interests, paper_title, paper_abstract):
    """I have the following research interests:\n {{research_interests}}.
    Based on the following paper determine if it is related to my research interests using your provided JSON schema.
    Paper title: {{paper_title}}
    Paper abstract: {{paper_abstract}}
    """
    pass


@prompt
def newsletter_prompt(title_abstract_content, research_interests, top_n_papers):
    """Use the following research paper content and my research interests in order to write me a personalized newsletter about the source material \
and how it's related to my research interests.

Research interests:
{{research_interests}}

Source material:
{{title_abstract_content}}

Write the newsletter in the provided JSON schema under draft. The draft should be a string.  Address the reader as "Dear Reader" and sign off as "PaperPal".
Write in a friendly and engaging tone and try to make the content flow together naturally. 
Ensure you covert all the papers you were provided in the source material you should have a total of {{top_n_papers}} papers discussed in the newsletter.
Do not hallucinate and do not make up information.
"""

@prompt
def research_prompt(research_interests, text):
    """I have the following research interests:
{{research_interests}}

Based on the content below delimited by <> please determine and output the following:

1. Determine if the research presented is related to any of my research interests.
2. Explain your rational of why this research is or is not related to my research interests with concrete reasons.
3. Provide a score between 1-10 with the following rubric:

Score 1-2: This research has no relevance to my research interests by topic or domain.
Score 3-4: This research has no relevance to my research insterests but might be of a similar domain.
Score 4-5: This research has relevance to at least one of my research interests.
Score 5-6: This research has relevance to more than one of my research interests.
Score 7-8: This research has relevance to all of my research interests or has relevance to more than one research interests but has a potentially ground breaking impact on the field.
Score 9-10: This research has relevance to all of my research interests and has a potentially ground breaking impact on the field.

<{{text}}>
"""
    pass