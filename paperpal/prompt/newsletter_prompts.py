from .prompting import prompt


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
Ensure you cover all the papers you were provided in the source material you should have a total of {{top_n_papers}} papers discussed in the newsletter.
Elaborate on each paper and explain to the reader why this paper is important, how it's exciting and how it is related to the research interests provided.
You are allowed to write in depth as this newsletter is meant to be a comprehensive summary of the latest research in a field.
Do not hallucinate and do not make up information.
Avoid discussing time and date as the date of the newsletter will be included in the email.
"""
    pass

@prompt
def newsletter_context_prompt(research_interests, title_abstract_content):
    """
Write a section of a personalized newsletter about the provided research paper, 
considering the given research interests.

Research Interests: 
{{research_interests}}

Source Material: 
{{title_abstract_content}}

Instructions:

*   **Provide** specific takeaways that the reader can take to learn more or apply the research presented in the paper.
*   **Include** a reflection on the research, such as how it relates to a common challenge or its potential impact.
*   **Elaborate** on the paper and explain to the reader why this paper is important and how it is exciting. 
*   **Write** in a friendly and engaging tone (e.g., "This fascinating study sheds light on..." or "Imagine the possibilities if...")
*   **Maintain** a concise and focused style.
*   **Do not** hallucinate or make up information.
*   **Output** Limit your response to no more than 200 words.

Constraints:

*   **Do not** include a salutation, signature, sections, headers, bullet points, or numbered lists.
*   **Do not** use phrases like "firstly", "secondly", etc.
*   **Do not** directly reference the research interests provided; they are for context only.
*   **Do not** say "Elaboration", "Actionable Advice", "Personal Reflection", or similar phrases.

    """
    pass


@prompt
def newsletter_final_prompt(content):
    """Finalize the newsletter draft using the following pre-written content:
CONTENT:
{{content}}

INSTRUCTIONS:
Revise the content to ensure that it's clean and doesn't have JSON or other formatting errors.
Your output should be identical to the input except for cleaning up the formatting to ensure that it is ready to be sent in an email.  Do not remove any content.

The content should be in the following order:
1. Newsletter Introduction
2. Newsletter Sections

Make sure that each section doesn't have the same introductory text (i.e. In this fascinating study...). Ensure that there is some variation in the introductory text.

Re-write the complete newsletter in the provided JSON schema under 'draft'.
Keep the newsletter in markdown format.
Do not hallucinate or make up information.
Avoid discussing time and date.
"""
    pass


@prompt
def newsletter_intro_prompt(sections):
    """You are writing the introduction for a newsletter. Below are sections written on research papers. Here is the content you are going to use to write the introduction:
PAPER CONTENT:
{% for section in sections %}
SECTION {{ loop.index }} START
{{ section }}
SECTION {{ loop.index }} END

{% endfor %}

INSTRUCTIONS:

*   **Format:** Address the reader as "Dear Reader" and sign off as "PaperPal". Do not use the section headers / footers that are in the Section Content. \
This is just context to help you write the introduction.
*   **Engagement:** Write an engaging introduction that sets the tone for the newsletter and draws the reader in and summarizes the key takeaways for the content in the paper content.
*   **Avoid:** Don't say "firstly", "secondly", "thirdly", lastly, etc. The introduction should be entirely standalone and not have any of these. Only write based off the information you are provided. \
Do not reference subjects or disciplines that aren't provided in the paper content.
Write the complete introduction in the provided JSON schema under 'draft'. 
Do not hallucinate or make up information.
Avoid discussing time and date.
"""
    pass


@prompt
def general_summary_prompt(query_content):
    """The following is a research paper extracted into markdown format. \
Do the following:
1.) Analyze the input text and generate 5 essential questions that, when answered, capture the main points and core meaning of the text.
2.) When formulating your questions: 
    a. Address the central theme or argument of the content
    b. Identify key supporting ideas 
    c. Highlight important facts or evidence 
    d. Reveal the author's purpose or perspective 
    e. Explore any significant implications or conclusions.
3.) Answer all of your generated questions one-by-one in detail in the content key.
4.) If the content is not relevant to the topic of interest or the plan, state that in the content key.

Paper:\n{{query_content}}"""
    pass


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