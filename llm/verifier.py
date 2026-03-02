"""
Verifier - Answer Quality Verification

Evaluates generated answers and provides structured feedback.
Supports multi-round verification with VERDICT parsing.
"""

import json
import re
from typing import Dict, Any, List, Optional
from llm.manager import LLMManager
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Inline verifier prompt (fallback if template not found)
VERIFIER_PROMPT = """You are a biomedical research verification expert. Evaluate the answer quality.

## Query
{query}

## Retrieved Context
{context}

## Generated Answer
{answer}

## Evaluation Criteria
1. **Factual Accuracy**: Is the answer supported by the context?
2. **Completeness**: Does it address all aspects of the query?
3. **Citation Quality**: Are claims properly attributed?
4. **Scientific Rigor**: Is the reasoning sound?

## Output Format
VERDICT: [PASS or FAIL]
CONFIDENCE: [0.0-1.0]
ISSUES:
- [issue description if any]
SUGGESTIONS:
- [improvement suggestion if any]
"""


class Verifier:
    """
    Verify quality of generated answers.

    Features:
    - VERDICT parsing (PASS/FAIL)
    - Confidence scoring
    - Structured issue detection
    - Multi-round verification support
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize Verifier.

        Args:
            llm_manager: LLM manager instance
        """
        self.llm = llm_manager

        logger.info("Verifier initialized")

    async def verify(
        self,
        query: str,
        context: str,
        answer: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Verify answer quality with VERDICT parsing.

        Args:
            query: Original query
            context: Retrieved context
            answer: Generated answer
            max_retries: Maximum verification retries

        Returns:
            Verification result dictionary:
            {
                "status": "PASS" | "FAIL",
                "confidence": 0.0-1.0,
                "issues": [...],
                "suggestions": [...],
                "raw_response": str
            }
        """
        logger.info(
            "Verifying answer",
            query=query[:100],
            answer_length=len(answer)
        )

        for attempt in range(max_retries):
            try:
                # Try template first, fallback to inline prompt
                try:
                    response = await self.llm.generate_with_template(
                        "verifier",
                        {
                            "query": query,
                            "context": context[:3000],
                            "answer": answer
                        },
                        temperature=0.4,
                        max_tokens=1000
                    )
                except Exception:
                    # Fallback to inline prompt
                    prompt = VERIFIER_PROMPT.format(
                        query=query,
                        context=context[:3000],
                        answer=answer
                    )
                    response = await self.llm.generate(
                        prompt,
                        temperature=0.4,
                        max_tokens=1000
                    )

                # Parse VERDICT from response
                result = self._parse_verifier_response(response)

                logger.info(
                    "Verification completed",
                    status=result["status"],
                    confidence=result.get("confidence"),
                    num_issues=len(result.get("issues", [])),
                    attempt=attempt + 1
                )

                return result

            except Exception as e:
                logger.warning(
                    "Verification attempt failed",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt == max_retries - 1:
                    # Return default PASS on final failure
                    return {
                        "status": "PASS",
                        "confidence": 0.7,
                        "issues": [],
                        "suggestions": [],
                        "error": str(e)
                    }

        return {"status": "PASS", "confidence": 0.7, "issues": [], "suggestions": []}

    def _parse_verifier_response(self, response: str) -> Dict[str, Any]:
        """
        Parse verifier output.

        Supports both:
        - JSON output (from config/prompts/verifier.j2)
        - VERDICT/CONFIDENCE/ISSUES/SUGGESTIONS text output (ev1 style / inline fallback)
        """
        json_result = self._try_parse_json_response(response)
        if json_result:
            return json_result
        return self._parse_verdict_response(response)

    def _try_parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Best-effort parse JSON verifier outputs (with optional code fences)."""
        if not response:
            return None

        text = response.strip()

        # Strip markdown code fences if present.
        fence_match = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        # If the model returned extra prose around JSON, try to extract the first {...} block.
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1].strip()

        if not text.startswith("{") or not text.endswith("}"):
            return None

        try:
            payload = json.loads(text)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        status_raw = str(payload.get("status", "")).strip().upper()
        status = "PASS" if status_raw == "PASS" else "FAIL" if status_raw == "FAIL" else "PASS"

        confidence = payload.get("confidence", 0.8)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.8
        confidence = min(1.0, max(0.0, confidence))

        issues_in = payload.get("issues", [])
        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []
        if isinstance(issues_in, list):
            for item in issues_in:
                if isinstance(item, dict):
                    issue_type = str(item.get("type", "verification_issue"))
                    desc = str(item.get("description", "")).strip()
                    sugg = str(item.get("suggestion", "")).strip()
                    if desc:
                        issues.append(
                            {
                                "type": issue_type or "verification_issue",
                                "description": desc,
                                "suggestion": sugg
                            }
                        )
                    if sugg:
                        suggestions.append(sugg)
                elif isinstance(item, str) and item.strip():
                    issues.append({"type": "verification_issue", "description": item.strip()})

        # If verifier returns FAIL with no concrete issues, keep minimal signal.
        if status == "FAIL" and not issues:
            issues = [
                {
                    "type": "verification_issue",
                    "description": "Verifier returned FAIL but did not provide structured issues.",
                    "suggestion": "Regenerate the answer by grounding claims more explicitly in the provided context."
                }
            ]

        # Deduplicate suggestions.
        suggestions = list(dict.fromkeys([s for s in suggestions if s]))

        return {
            "status": status,
            "confidence": confidence,
            "issues": issues,
            "suggestions": suggestions,
            "raw_response": response
        }

    def _parse_verdict_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VERDICT from verifier response.

        Handles both structured and free-form responses.

        Args:
            response: Raw verifier response

        Returns:
            Parsed verification result
        """
        result = {
            "status": "PASS",
            "confidence": 0.8,
            "issues": [],
            "suggestions": [],
            "raw_response": response
        }

        response_upper = response.upper()

        # Parse VERDICT
        verdict_match = re.search(r'VERDICT\s*[:\-]?\s*(PASS|FAIL)', response_upper)
        if verdict_match:
            result["status"] = verdict_match.group(1)
        else:
            # Fallback: look for PASS/FAIL anywhere
            if "FAIL" in response_upper and "PASS" not in response_upper:
                result["status"] = "FAIL"
            elif "VERDICT: FAIL" in response_upper or "VERDICT:FAIL" in response_upper:
                result["status"] = "FAIL"

        # Parse CONFIDENCE
        confidence_match = re.search(r'CONFIDENCE\s*[:\-]?\s*([\d.]+)', response_upper)
        if confidence_match:
            try:
                conf = float(confidence_match.group(1))
                result["confidence"] = min(1.0, max(0.0, conf))
            except ValueError:
                pass
        else:
            # Default confidence based on status
            result["confidence"] = 0.85 if result["status"] == "PASS" else 0.4

        # Parse ISSUES
        issues_section = re.search(r'ISSUES?\s*[:\-]?\s*\n(.*?)(?=SUGGESTION|$)', response, re.DOTALL | re.IGNORECASE)
        if issues_section:
            issue_lines = issues_section.group(1).strip().split('\n')
            for line in issue_lines:
                line = line.strip().lstrip('-').lstrip('•').strip()
                if line and len(line) > 5:
                    result["issues"].append({
                        "type": "verification_issue",
                        "description": line
                    })

        # Parse SUGGESTIONS
        suggestions_section = re.search(r'SUGGESTIONS?\s*[:\-]?\s*\n(.*?)$', response, re.DOTALL | re.IGNORECASE)
        if suggestions_section:
            suggestion_lines = suggestions_section.group(1).strip().split('\n')
            for line in suggestion_lines:
                line = line.strip().lstrip('-').lstrip('•').strip()
                if line and len(line) > 5:
                    result["suggestions"].append(line)

        return result

    async def verify_with_regeneration_prompt(
        self,
        query: str,
        context: str,
        answer: str,
        verification_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a regeneration prompt based on verification issues.

        Used when verification FAILs to guide answer improvement.

        Args:
            query: Original query
            context: Context
            answer: Failed answer
            verification_result: Verification result with issues

        Returns:
            Regeneration prompt or None if no changes needed
        """
        if verification_result["status"] == "PASS":
            return None

        issues = verification_result.get("issues", [])
        suggestions = verification_result.get("suggestions", [])

        if not issues and not suggestions:
            return "The previous answer was evaluated as FAIL but no specific issues were identified. Please review the context carefully and regenerate a more accurate and rigorous answer."

        prompt_parts = [
            "The previous answer had issues that need to be addressed:",
            ""
        ]

        for issue in issues:
            desc = issue.get("description", str(issue))
            prompt_parts.append(f"- {desc}")

        if suggestions:
            prompt_parts.append("")
            prompt_parts.append("Suggestions for improvement:")
            for s in suggestions:
                prompt_parts.append(f"- {s}")

        prompt_parts.append("")
        prompt_parts.append("Please regenerate a better answer addressing these issues.")

        return "\n".join(prompt_parts)

    def get_suggestions(self, verification_result: Dict[str, Any]) -> List[str]:
        """
        Extract actionable suggestions from verification result.

        Args:
            verification_result: Result from verify()

        Returns:
            List of suggestion strings
        """
        suggestions = verification_result.get("suggestions", [])

        # Also extract suggestions from issues
        for issue in verification_result.get("issues", []):
            if isinstance(issue, dict) and issue.get("suggestion"):
                suggestions.append(issue["suggestion"])

        return list(set(suggestions))  # Deduplicate

    def should_regenerate(self, verification_result: Dict[str, Any]) -> bool:
        """
        Determine if answer should be regenerated based on verification.

        Args:
            verification_result: Result from verify()

        Returns:
            True if regeneration is needed
        """
        if verification_result["status"] == "FAIL":
            return True

        # Also check confidence threshold
        confidence = verification_result.get("confidence", 1.0)
        if confidence < 0.5:
            return True

        return False
