from prompting import prompt

@prompt
def wizard_vicuna_prompt(context, research_interests):
    """USER: I have the following research interests:
{{ research_interests }}

Based on the content below delimited by <> does this paper directly relate to any of my research interests?
Respond in a json with the keys related (bool) and reasoning (str).

<{{ context }}>
ASSISTANT:"""
    pass


@prompt
def vicuna_prompt(context, research_interests):
    """SYSTEM:A chat between a curious human and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the human's questions.
HUMAN: I have the following research interests:
{{ research_interests }}

Based on the content below delimited by <> does this paper directly relate to any of my research interests?
Respond in a json with the keys related (bool) and reasoning (str).

<{{ context }}>
ASSISTANT:"""
    pass


@prompt
def openchat_prompt(context, research_interests):
    """GPT4 Correct User: I have the following research interests:
{{ research_interests }}
Based on the content below delimited by <> does this paper directly relate to any of my research interests?
Respond in a json with the keys related (bool) and reasoning (str).

<{{ context }}><|end_of_turn|>GPT4 Correct Assistant:
"""
    pass


@prompt
def zephyr_prompt(context, research_interests):
    """# <|system|> You are an AI assistant who always follows instructions.</s>
<|user|> I have the following research interests:
{{ research_interests }}
Based on the content below delimited by <> does this paper directly relate to any of my research interests?
Respond in a json with the keys related (bool) and reasoning (str).

<{{ context }}></s>
<|assistant|>"""
    pass