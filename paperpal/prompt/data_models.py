from pydantic import BaseModel

class ResearchInterestsPromptData(BaseModel):
    related: bool
    rationale: str
    score: int

class NewsletterPromptData(BaseModel):
    draft: str

class SummaryPromptData(BaseModel):
    questions: str
    content: str

