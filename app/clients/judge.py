import json
from dataclasses import dataclass

from anthropic import Anthropic

from app.utils.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


@dataclass
class JudgeResult:
    score: int | None
    rationale: str | None
    error: str | None


class AnthropicJudge:
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def score(self, transcript_text: str, qa_report_json: str) -> JudgeResult:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(
                            transcript_text=transcript_text,
                            qa_report_json=qa_report_json,
                        ),
                    }
                ],
            )
            parsed = json.loads(response.content[0].text)
            return JudgeResult(score=parsed["score"], rationale=parsed["rationale"], error=None)
        except Exception as e:
            return JudgeResult(score=None, rationale=None, error=str(e))
