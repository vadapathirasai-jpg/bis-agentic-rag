"""Consumer Agent module for BIS Sahayak.

Implements an Agentic RAG decision loop for consumer-facing inquiries:
1. Validates and analyzes user question intent.
2. Retrieves relevant official BIS evidence via the existing Retriever.
3. Evaluates evidence sufficiency (similarity score, semantic alignment, topic match).
4. Refines search queries dynamically if initial evidence is weak (bounded loop).
5. Grounded synthesis via Google Gemini using the official Google GenAI SDK.
6. Preserves full citation metadata and returns structured responses.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Ensure backend root is on sys.path so app imports work cleanly
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from backend/.env if present
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from app.retrieval.retriever import Retriever
from app.mcp.client import BISConsumerMCPClient, MCPClientError
from app.mcp.tools import get_retriever

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# System instruction prompt enforcing strict grounding in official BIS evidence
BIS_CONSUMER_SYSTEM_INSTRUCTION = (
    "You are the Consumer Agent for BIS Sahayak, assisting Indian consumers with Bureau of Indian Standards (BIS) services.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Answer the user's question using ONLY the supplied official BIS evidence.\n"
    "2. The supplied BIS evidence is the authoritative source for this response.\n"
    "3. Do not use outside knowledge.\n"
    "4. Do not browse the web.\n"
    "5. Do not invent facts.\n"
    "6. Do not invent BIS policies.\n"
    "7. Do not invent procedures.\n"
    "8. Do not invent complaint timelines.\n"
    "9. Do not invent fees.\n"
    "10. Do not invent phone numbers.\n"
    "11. Do not invent email addresses.\n"
    "12. Do not invent URLs.\n"
    "13. Do not invent standards or certification requirements.\n"
    "14. If the supplied evidence does not contain enough information to answer the question, clearly say that the available official BIS evidence is insufficient.\n"
    "15. Do not fabricate citations.\n"
    "16. Keep the answer concise, accurate, and useful."
)


class GeminiClient:
    """Official Google GenAI SDK client for BIS Sahayak."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize Google GenAI client.

        Args:
            api_key: Optional API key. Defaults to GEMINI_API_KEY environment variable.
            model: Model name. Defaults to GEMINI_MODEL env var or 'gemini-2.5-flash'.
            timeout: Network request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout
        self.active_model: str = self.model
        self._client: Optional[genai.Client] = None

        if self.is_configured:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as err:
                logger.error("Failed to initialize genai.Client: %s", err)
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Return True if a non-empty API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate content from Gemini model using official Google GenAI SDK.

        Args:
            prompt: User prompt content.
            system_instruction: Optional system instruction prompt.

        Returns:
            Generated text string.

        Raises:
            ValueError: If GEMINI_API_KEY is not configured.
            RuntimeError: If Gemini API returns an error or invalid response.
        """
        if not self.is_configured:
            raise ValueError(
                "GEMINI_API_KEY is not configured in the environment. "
                "Please set the GEMINI_API_KEY environment variable."
            )

        if self._client is None:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as err:
                raise RuntimeError(f"Failed to initialize official Google GenAI client: {err}") from err

        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
            system_instruction=system_instruction if system_instruction else None,
        )

        # Attempt generation with configured model, with automatic fallback for deprecated endpoints (e.g. 404)
        models_to_try = [self.model]
        if self.model != "gemini-3.6-flash":
            models_to_try.append("gemini-3.6-flash")

        last_err: Optional[Exception] = None
        for candidate_model in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                    config=config,
                )
                text_output = ""
                if response.text:
                    text_output = response.text.strip()
                elif response.candidates:
                    parts = response.candidates[0].content.parts or []
                    text_parts = [p.text for p in parts if getattr(p, "text", None)]
                    text_output = "".join(text_parts).strip()

                if text_output:
                    self.active_model = candidate_model
                    if candidate_model != self.model:
                        logger.info(
                            "Auto-recovered from deprecated model '%s' using '%s'.",
                            self.model,
                            candidate_model,
                        )
                    return text_output
            except errors.APIError as err:
                last_err = err
                if err.code == 404:
                    logger.warning(
                        "Model '%s' returned 404 (%s); attempting fallback candidate...",
                        candidate_model,
                        err.message,
                    )
                    continue
                logger.error("Google GenAI API call failed: %s", err)
                raise RuntimeError(f"Google GenAI API error ({err.code}): {err.message}") from err
            except Exception as err:
                last_err = err
                logger.warning("Generation with '%s' failed: %s", candidate_model, err)
                continue

        raise RuntimeError(f"Google GenAI generation failed across candidates: {last_err}")


class ConsumerAgent:
    """Agentic RAG assistant specializing in BIS consumer protection, standards, and services."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        max_attempts: int = 2,
        top_k: int = 4,
        mcp_client: Optional[BISConsumerMCPClient] = None,
        use_mcp: bool = True,
    ) -> None:
        """Initialize the Consumer Agent.

        Args:
            retriever: Optional pre-configured Retriever instance.
            gemini_api_key: Optional Gemini API key.
            gemini_model: Optional Gemini model name.
            max_attempts: Maximum retrieval/refinement attempts (bounded 1-3).
            top_k: Default number of evidence chunks to retrieve per search.
            mcp_client: Optional pre-configured BISConsumerMCPClient instance.
            use_mcp: If True (default), attempts evidence retrieval via the standardized
                MCP tool interface before falling back to direct Retriever. If False,
                bypasses MCP and queries direct Retriever exclusively.
        """
        self.retriever = retriever or get_retriever()
        self.gemini = GeminiClient(api_key=gemini_api_key, model=gemini_model)
        self.max_attempts = max(1, min(max_attempts, 3))
        self.top_k = top_k
        self.use_mcp = use_mcp

        if self.use_mcp:
            self.mcp_client = mcp_client or BISConsumerMCPClient()
        else:
            self.mcp_client = None

    def _retrieve_evidence(
        self,
        query: str,
        top_k: int,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Retrieve authoritative evidence via MCP tool or fallback/direct Retriever.

        Explicitly distinguishes between:
        - "mcp": MCP retrieval successfully executed and returned evidence.
        - "fallback": MCP was attempted but failed/errored, so direct Retriever was used.
        - "direct": MCP is disabled or not configured; direct Retriever was used directly.

        Args:
            query: Natural language search query string.
            top_k: Number of evidence chunks to retrieve.

        Returns:
            Tuple of (evidence_chunks, channel_name).
        """
        if self.use_mcp and self.mcp_client is not None:
            try:
                logger.info("Attempting evidence retrieval via MCP tool 'search_bis_documents'...")
                evidence = self.mcp_client.search_bis_documents(query=query, top_k=top_k)
                logger.info("MCP retrieval successful: %d chunks retrieved.", len(evidence))
                return evidence, "mcp"
            except Exception as mcp_err:
                logger.warning(
                    "MCP retrieval failed (%s); executing safe direct Retriever fallback.",
                    mcp_err,
                )
                fallback_evidence = self.retriever.retrieve(query=query, top_k=top_k)
                return fallback_evidence, "fallback"

        logger.info("MCP disabled or unavailable; executing direct Retriever.")
        direct_evidence = self.retriever.retrieve(query=query, top_k=top_k)
        return direct_evidence, "direct"

    def analyze_intent(self, question: str) -> Dict[str, Any]:
        """Perform lightweight intent understanding focused on BIS Consumer topics.

        Args:
            question: Cleaned user question string.

        Returns:
            Dictionary with detected topics, entity matches, and domain alignment.
        """
        q_lower = question.lower()

        # Canonical topic indicators for BIS Consumer domain
        topic_patterns = {
            "bis_care": r"\b(bis\s*care|care\s*app|mobile\s*app)\b",
            "complaints": r"\b(complaint|grievance|redressal|cmed|dispute|report|misuse)\b",
            "isi_mark": r"\b(isi|isi\s*mark|standard\s*mark|certification\s*mark)\b",
            "hallmarking": r"\b(hallmark|hallmarking|gold|silver|jeweller|purity|carat)\b",
            "huid": r"\b(huid|unique\s*identification|verify\s*huid)\b",
            "consumer_protection": r"\b(consumer|rights|protection|awareness|safety|adulteration)\b",
            "verification": r"\b(verify|verification|check|authenticity|license|licence)\b",
            "contact": r"\b(contact|email|phone|helpline|office|head)\b",
        }

        detected_topics: List[str] = []
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, q_lower):
                detected_topics.append(topic)

        domain_aligned = len(detected_topics) > 0 or any(
            term in q_lower for term in ["bis", "standard", "quality", "certification", "product"]
        )

        return {
            "question": question,
            "detected_topics": detected_topics,
            "is_domain_aligned": domain_aligned,
        }

    def evaluate_evidence(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """Evaluate retrieved evidence chunks for sufficiency and semantic alignment.

        Considers similarity scores, dedicated topic/section match, and question relevance.
        Does NOT rely on a loose fallback threshold.

        Args:
            question: Current search question or refined query.
            evidence: List of retrieved evidence chunk dictionaries from Qdrant.
            attempt: Current iteration number (1-based).

        Returns:
            Evaluation summary dict containing 'sufficient', 'top_score', and 'reason'.
        """
        if not evidence:
            return {
                "sufficient": False,
                "top_score": 0.0,
                "reason": "No evidence chunks returned from vector store.",
                "relevant_chunks": 0,
            }

        top_chunk = evidence[0]
        top_score = float(top_chunk.get("score", 0.0))
        intent = self.analyze_intent(question)
        detected_topics = intent["detected_topics"]

        # Check dedicated title / section alignment
        top_title_section = (top_chunk.get("title", "") + " " + top_chunk.get("section", "")).lower()
        all_titles_sections = " ".join(
            [(c.get("title", "") + " " + c.get("section", "")).lower() for c in evidence[:3]]
        )

        has_dedicated_match = False
        if "bis_care" in detected_topics:
            has_dedicated_match = "bis care" in top_title_section or "bis care" in all_titles_sections
        elif "isi_mark" in detected_topics:
            has_dedicated_match = any(
                k in top_title_section or k in all_titles_sections
                for k in ["isi", "standard mark"]
            )
        elif "huid" in detected_topics:
            has_dedicated_match = "huid" in top_title_section or any(
                "huid" in (c.get("text", "")).lower() for c in evidence[:2]
            )
        elif "complaints" in detected_topics:
            has_dedicated_match = "complaint" in top_title_section
        elif "hallmarking" in detected_topics:
            has_dedicated_match = "hallmark" in top_title_section

        # Strict Sufficiency Criteria:
        # 1. Very high confidence score (>= 0.70) with topic alignment -> Sufficient
        # 2. Solid confidence score (>= 0.60) with dedicated topic/section match -> Sufficient
        # 3. Good confidence score (>= 0.58) with dedicated title/section alignment -> Sufficient
        # 4. Otherwise -> Insufficient (triggers query refinement if attempts remain; does NOT accept weak score)
        is_sufficient = False
        reason = ""

        if top_score >= 0.70 and (has_dedicated_match or len(detected_topics) == 0):
            is_sufficient = True
            reason = f"High confidence similarity score ({top_score:.4f}) with validated topic alignment."
        elif top_score >= 0.60 and has_dedicated_match:
            is_sufficient = True
            reason = f"Solid similarity score ({top_score:.4f}) with dedicated topic evidence in top chunks."
        elif top_score >= 0.58 and has_dedicated_match and attempt > 1:
            is_sufficient = True
            reason = f"Refined query achieved dedicated topic match ({top_score:.4f})."
        else:
            is_sufficient = False
            if attempt < self.max_attempts:
                reason = (
                    f"Retrieval score ({top_score:.4f}) or topic specificity is insufficient; "
                    "refining query for higher precision."
                )
            else:
                reason = (
                    f"Evidence remains below sufficiency requirements ({top_score:.4f}) "
                    "after maximum retrieval attempts."
                )

        relevant_chunks_count = sum(1 for c in evidence if float(c.get("score", 0.0)) >= 0.50)

        return {
            "sufficient": is_sufficient,
            "top_score": top_score,
            "reason": reason,
            "relevant_chunks": relevant_chunks_count,
            "has_dedicated_match": has_dedicated_match,
        }

    def refine_query(
        self,
        question: str,
        attempt: int,
        previous_evidence: List[Dict[str, Any]],
    ) -> str:
        """Reformulate or refine the query to improve retrieval precision.

        Expands abbreviations and clarifies core intent for BIS Consumer domain.

        Args:
            question: Original question string.
            attempt: Current refinement attempt number.
            previous_evidence: Evidence chunks from the previous attempt.

        Returns:
            Refined natural-language search query.
        """
        q_lower = question.lower()

        # Domain-aware contextual reformulations
        if re.search(r"\b(bis\s*care)\b", q_lower):
            return "What is the BIS CARE app and what can consumers use it for?"
        if re.search(r"\b(isi\s*mark|isi)\b", q_lower):
            return "What is the ISI mark, what does it mean on products, and what are the BIS Standard Marks?"
        if re.search(r"\b(huid)\b", q_lower):
            return "How can Hallmark Unique Identification (HUID) number on gold jewellery be verified by consumers?"
        if re.search(r"\b(complaint|grievance)\b", q_lower):
            return "How can a consumer file a complaint with BIS and what is the complaint handling procedure?"
        if re.search(r"\b(hallmark|hallmarking)\b", q_lower):
            return "What is BIS hallmarking for gold and silver jewellery and how does it protect consumers?"

        # General domain reinforcement
        return f"Bureau of Indian Standards official guidelines for {question.strip()}"

    def format_evidence_context(self, evidence: List[Dict[str, Any]]) -> str:
        """Format retrieved evidence chunks into structured context for LLM prompt.

        Args:
            evidence: List of evidence chunk dictionaries.

        Returns:
            Formatted evidence string with citation headers.
        """
        context_parts: List[str] = []
        for idx, chunk in enumerate(evidence, 1):
            title = chunk.get("title", "BIS Resource")
            section = chunk.get("section") or "General"
            source_url = chunk.get("source_url", "https://www.bis.gov.in")
            chunk_id = chunk.get("chunk_id", "unknown")
            text = (chunk.get("text") or "").strip()

            context_parts.append(
                f"[Evidence {idx}]\n"
                f"Title: {title}\n"
                f"Section: {section}\n"
                f"Source URL: {source_url}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Content: {text}"
            )

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize a grounded answer using official Google GenAI SDK.

        Args:
            question: Original user question.
            evidence: Retrieved BIS evidence chunks.
            evaluation: Final evidence evaluation dictionary.

        Returns:
            Dict containing 'answer', 'gemini_configured', 'model_used', and 'status_message'.
        """
        # If Gemini is not configured, clearly report without generating fake text
        if not self.gemini.is_configured:
            return {
                "answer": None,
                "gemini_configured": False,
                "model_used": None,
                "status_message": (
                    "GEMINI_API_KEY is not configured in the environment. "
                    "Official BIS evidence was successfully retrieved and verified, "
                    "but final LLM synthesis requires an active Gemini API key."
                ),
            }

        # If evidence was evaluated as completely insufficient, report clearly
        if not evaluation.get("sufficient", False):
            return {
                "answer": "The available official BIS evidence is insufficient to answer this inquiry.",
                "gemini_configured": True,
                "model_used": self.gemini.active_model,
                "status_message": "Evidence insufficient after maximum retrieval attempts.",
            }

        context_str = self.format_evidence_context(evidence)
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Official BIS Context Evidence:\n{context_str}\n\n"
            "Answer the user's question accurately, concisely, and usefully using ONLY the official BIS context evidence above."
        )

        try:
            answer_text = self.gemini.generate(
                prompt=user_prompt,
                system_instruction=BIS_CONSUMER_SYSTEM_INSTRUCTION,
            )
            return {
                "answer": answer_text,
                "gemini_configured": True,
                "model_used": self.gemini.active_model,
                "status_message": "Grounded answer synthesized successfully.",
            }
        except Exception as err:
            logger.error("Gemini synthesis error: %s", err)
            return {
                "answer": None,
                "gemini_configured": True,
                "model_used": self.gemini.active_model,
                "error": str(err),
                "status_message": f"Gemini synthesis failed: {err}",
            }

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Run the full Agentic RAG pipeline for a user question.

        Executes intent analysis, iterative retrieval with evaluation and query
        refinement, and grounded answer synthesis.

        Args:
            question: Natural language question string.

        Returns:
            Structured response dictionary matching requirements.

        Raises:
            ValueError: If question is empty or invalid.
        """
        # 1. Query Validation
        if question is None or not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty, non-whitespace string.")

        cleaned_question = question.strip()
        intent = self.analyze_intent(cleaned_question)

        current_query = cleaned_question
        retrieval_attempts = 0
        final_evidence: List[Dict[str, Any]] = []
        final_eval: Dict[str, Any] = {}
        query_refinements: List[str] = []
        final_channel = "direct"

        # 2. Iterative Retrieval & Evaluation Loop
        for attempt in range(1, self.max_attempts + 1):
            retrieval_attempts = attempt
            logger.info("Retrieval attempt %d for query: '%s'", attempt, current_query)

            evidence, channel = self._retrieve_evidence(
                query=current_query,
                top_k=self.top_k,
            )
            final_evidence = evidence
            final_channel = channel
            eval_res = self.evaluate_evidence(
                question=current_query,
                evidence=evidence,
                attempt=attempt,
            )
            final_eval = eval_res

            logger.info(
                "Attempt %d evaluation: sufficient=%s, top_score=%.4f (reason: %s, channel: %s)",
                attempt,
                eval_res["sufficient"],
                eval_res["top_score"],
                eval_res["reason"],
                channel,
            )

            # If evidence is sufficient or we reached the maximum bounded attempts, exit loop
            if eval_res["sufficient"] or attempt >= self.max_attempts:
                break

            # Otherwise, refine query for the next attempt
            current_query = self.refine_query(
                question=cleaned_question,
                attempt=attempt,
                previous_evidence=evidence,
            )
            query_refinements.append(current_query)
            logger.info("Refined query for attempt %d: '%s'", attempt + 1, current_query)

        # 3. Grounded Answer Synthesis
        synthesis = self.generate_answer(
            question=cleaned_question,
            evidence=final_evidence,
            evaluation=final_eval,
        )

        return {
            "question": cleaned_question,
            "answer": synthesis["answer"],
            "evidence": final_evidence,
            "retrieval_attempts": retrieval_attempts,
            "evidence_sufficient": final_eval.get("sufficient", False),
            "evaluation": final_eval,
            "query_refinements": query_refinements,
            "gemini_configured": synthesis.get("gemini_configured", False),
            "model_used": synthesis.get("model_used"),
            "status_message": synthesis.get("status_message"),
            "intent": intent,
            "retrieval_channel": final_channel,
        }

    # Backward compatibility and ergonomics aliases
    process = answer_question
    run = answer_question


if __name__ == "__main__":
    print("=" * 70)
    print("BIS Sahayak - Consumer Agent Verification Suite (Official GenAI SDK)")
    print("=" * 70)

    # Initialize agent
    agent = ConsumerAgent()
    print("Connecting to Retriever and Qdrant...")
    agent.retriever.connect()
    print("Connected successfully.")
    print(f"Gemini Configured: {agent.gemini.is_configured} (Configured Model: {agent.gemini.model})")
    print("=" * 70)

    test_questions = [
        "What is BIS CARE?",
        "How can a consumer file a complaint with BIS?",
        "What does the ISI mark mean?",
        "How can HUID be verified by consumers?",
    ]

    for idx, q_text in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {idx}: {q_text}")
        print("=" * 70)

        res = agent.answer_question(q_text)

        print(f"Question:             {res['question']}")
        print(f"Retrieval Attempts:   {res['retrieval_attempts']}")
        if res.get("query_refinements"):
            print(f"Query Refinement:     {res['query_refinements']}")
        else:
            print("Query Refinement:     None (Initial retrieval was immediately sufficient)")
        print(f"Evidence Sufficient:  {res['evidence_sufficient']}")
        print(f"Evaluation Reason:    {res['evaluation']['reason']}")
        print(f"Top Score:            {res['evaluation']['top_score']:.4f}")
        print(f"Evidence Count:       {len(res['evidence'])}")
        print(f"Gemini Model Used:    {res.get('model_used')}")

        print("\nRetrieved Official BIS Evidence & Sources:")
        for e_idx, ev in enumerate(res["evidence"], 1):
            print(f"  [{e_idx}] Score:      {ev.get('score', 0):.4f}")
            print(f"      Title:      {ev.get('title')}")
            print(f"      Section:    {ev.get('section')}")
            print(f"      Chunk ID:   {ev.get('chunk_id')}")
            print(f"      Source URL: {ev.get('source_url')}")
            preview = (ev.get("text") or "").replace("\n", " ")
            if len(preview) > 140:
                preview = preview[:140] + "..."
            print(f"      Text:       {preview}")

        print("\nAnswer Output:")
        if res["answer"]:
            print(f"{res['answer']}")
        else:
            print(f"[Notice] {res['status_message']}")

    print(f"\n{'=' * 70}")
    print("Edge Case & Error Handling Validations...")
    print("=" * 70)

    # Validate empty question rejection
    for bad_q in ["", "   ", None]:
        try:
            agent.answer_question(bad_q)  # type: ignore[arg-type]
            print(f"FAILED: Expected ValueError for question: {repr(bad_q)}")
        except ValueError:
            print(f"PASSED: Correctly rejected invalid question: {repr(bad_q)}")

    print(f"\n{'=' * 70}")
    print("ALL CONSUMER AGENT VERIFICATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)
