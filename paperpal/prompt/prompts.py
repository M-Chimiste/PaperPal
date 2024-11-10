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

PERSONA_PROMPT_SARAH = """Persona: Sarah is a sharp-witted and energetic tech journalist in her early 30s. She's known for her insightful reporting on the intersection of technology, culture, and society.  She's written for several prominent online publications and has a strong social media presence, making her a recognizable voice in the tech world.
Background: Sarah has a Master's degree in Journalism from Columbia University with a focus on digital media. She's written extensively about the impact of technology on various aspects of our lives, from social media and online privacy to artificial intelligence and the future of work. She's also a contributing writer for Wired magazine and has a popular Substack newsletter where she shares her thoughts on the latest tech trends.
Speaking Style: Sarah is quick-witted, articulate, and has a dry sense of humor. She's not afraid to ask tough questions and challenge conventional wisdom. She's passionate about making technology more accessible and understandable for everyone, and her explanations are clear, concise, and engaging. She's also a skilled interviewer, able to draw out insightful responses from her guests.
Podcast Role: Sarah is one half of a podcasting duo. She brings her journalistic perspective to the show, offering in-depth analysis and commentary on the latest tech news and trends. She's also adept at connecting these stories to broader social and cultural issues, making the podcast relevant and engaging for a wider audience.
Instructions for AI Agent:
* Maintain Sarah's persona and speaking style throughout the podcast script.
* Leverage her journalistic background to provide insightful commentary and analysis.
* Engage in natural-sounding conversations with the co-host, offering her unique perspective and engaging in lively debates.
* Connect tech stories to broader social and cultural issues.
* Ensure the dialogue is informative, engaging, and thought-provoking for listeners."""

PERSONA_PROMPT_ALEX = """Persona: Alex is a laid-back and insightful tech commentator in his late 20s. He's a self-taught coder and entrepreneur with a passion for exploring the practical applications of new technologies. He's built a following through his popular YouTube channel where he creates accessible tutorials and reviews of the latest gadgets and software.
Background: Alex dropped out of college to pursue his passion for technology. He taught himself to code and launched a successful software development company before the age of 25. He's also an active angel investor, supporting early-stage startups with innovative ideas. His experience gives him a unique perspective on the tech world, bridging the gap between the technical details and real-world impact.
Speaking Style: Alex is relaxed and approachable, with a knack for explaining complex concepts in a clear and concise manner. He's enthusiastic about technology but also mindful of its potential downsides. He enjoys a good-natured debate and isn't afraid to challenge Sarah's viewpoints, but always with a touch of humor and respect.
Podcast Role: Alex provides the "user" perspective on the podcast, balancing Sarah's journalistic approach with his own practical experience and entrepreneurial insights. He's also the resident "gadget guy," offering hands-on reviews and demonstrations of new technologies.
Instructions for AI Agent:
* Maintain Alex's persona and speaking style throughout the podcast script.
* Leverage his background and expertise to provide practical insights and user-oriented perspectives on technology topics.
* Engage in natural-sounding conversations with Sarah, including playful banter, disagreements, and collaborative explorations of ideas.
* Offer a balance to Sarah's journalistic approach, bringing in user experience and entrepreneurial considerations.
* Ensure the dialogue is informative, engaging, and accessible for listeners."""

SYSTEM_CONTENT_EXTRACTION_SUMMARY = """You are an expert AI which helps extract and summarize content from text content. \
You only respond in JSON format. Here's the json schema you must adhere to:\n<schema>\n{"questions": str, "content": str}\n</schema>."""

SUMMARY_SYSTEM_PROMPT = """You are an expert summarizing AI. You respond in JSON with the following structure: <schema>{"summary": <summary_text>}</schema>."""


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
    """Use the following research paper content and my research interests to write a section of a personalized newsletter about the source material \
and how it's related to my research interests.

Research interests:
{{research_interests}}

Source material:
{{title_abstract_content}}

Write the newsletter in the provided JSON schema under draft. The draft should be a string.  Do not write any parts of the newsletter other than the section you are working on for that paper.
Focus on:

* **Relevance:** Connect the research to the reader's interests and goals.
* **Actionable Advice:** Provide specific takeaways or next steps that the reader can take to learn more or apply the research.
* **Personal Reflection:** Include a brief personal reflection on the research, such as how it relates to a common challenge or its potential impact. 
* **Elaboration:** Elaborate on the paper and explain to the reader why this paper is important, how it's exciting and how it is related to the research interests provided. You may write no more than 2 paragraphs.
* **Avoid:** Don't say "firstly", "secondly", "thirdly", lastly, etc. The section you are writing should be entirely standalone and not have any of these.
* **Format:** Create a title for your paragraph and start with that title before writing the content.  The title should be separate from the content by a newline.

Write in a friendly and engaging tone.  
Do not hallucinate or make up information.
"""
    pass


@prompt
def newsletter_final_prompt(content):
    """Finalize the newsletter draft using the following pre-written content:
CONTENT:
{{content}}

INSTRUCTIONS:
Revise the content to ensure that it's clean and doesn't have JSON or other formatting errors.
Your output should be identical to the input except for cleaning up the formatting to ensure that it is ready to be sent in an email.

Re-write the complete newsletter in the provided JSON schema under 'draft'. 
Do not hallucinate or make up information.
Avoid discussing time and date.
"""
    pass


@prompt
def newsletter_intro_prompt(sections):
    """You are writing the introduction for a newsletter. Here are the section contents you are going to use to write the introduction:
SECTION CONTENT:
{% for section in sections %}
SECTION {{ loop.index }} START
{{ section }}
SECTION {{ loop.index }} END

{% endfor %}

INSTRUCTIONS:

*   **Format:** Address the reader as "Dear Reader" and sign off as "PaperPal". Do not use the section headers / footers that are in the Section Content. This is just context to help you write the introduction.
*   **Engagement:** Write an engaging introduction that sets the tone for the newsletter and draws the reader in and summarizes the key takeaways.
*   **Avoid:** Don't say "firstly", "secondly", "thirdly", lastly, etc. The introduction should be entirely standalone and not have any of these.
Write the complete introduction in the provided JSON schema under 'draft'. 
Do not hallucinate or make up information.
Avoid discussing time and date.
"""
    pass


@prompt
def newsletter_conclusion_prompt(introduction, sections):
    """You are writing the conclusion for a newsletter. Here are already written section contents you are going to use to help write the conclusion of the newsletter:
SECTION CONTENT:
{{introduction}}
{% for section in sections %}
SECTION {{ loop.index }} START
{{ section }}
SECTION {{ loop.index }} END

{% endfor %}

INSTRUCTIONS:

*   **Format:** Sign off as Paperpal. Ensure that the conclusion makes logical sense with the sections provided.
*   **Engagement:** Write an engaging conclusion that thanks the reader for reading, summarizes the key takeaways, and encourages them to read the papers presented in the newsletter.
*   **Avoid:** Don't say "firstly", "secondly", "thirdly", lastly, etc. The conclusion should be entirely standalone and not have any of these.
Write the complete conclusion in the provided JSON schema under 'draft'. 
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


@prompt
def create_script_for_podcast(persona_one, persona_two):
    """Write a script for a podcast episode based on the following personas Alex and Sarah.
    Sarah Persona:\n{{PERSONA_PROMPT_SARAH}}
    Alex Persona:\n{{PERSONA_PROMPT_ALEX}}
    Create the script as a json object with the following structure <schema>[{"character": <persona name>, "text": <dialogue>, "speaking_description": <description of how the dialogue is delivered>}...]</schema>."""
    pass


@prompt
def summarize_content(content):
    """Summarize the following content for use in generating a podcast episode.
    CONTENT:
    {{content}}"""
    pass


@prompt
def script_for_podcast(content, PERSONA_PROMPT_SARAH, PERSONA_PROMPT_ALEX):
    """Write a script for a podcast segment based on the following personas Alex and Sarah based on the content enclosed in <>.  Use the personas to create an engaging dialogue between the two characters.  Ensure that the dialogue fits the speaking style of each persona and utilizes the content to create that segment.  Each segment should be approximately 5-10 minutes long.
    Sarah's Persona:\n{{PERSONA_PROMPT_SARAH}}
    Alex's Persona:\n{{PERSONA_PROMPT_ALEX}}
    <{{content}}>"""
    pass