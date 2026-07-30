import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from app.retrieval import retrieve_policy_chunks
from app.schemas import PolicyResponse
from app.citations import extract_citations, validate_citations, build_citations

MIN_CONFIDENCE_SCORE = 0.25
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SOURCE_INDEX_PATTERN = re.compile(r"\[SOURCE:(\d+)\]")
REFUSE_TOKEN_PATTERN = re.compile(r"(?<!\w)REFUSE(?!\w)[.:,]?\s*")
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

POLICY_PROMPT = PromptTemplate(
    input_variables=["question", "sources"],
    template="""You are a policy compliance assistant for Google Ads.

Answer using ONLY the sources below. Every factual claim MUST include a citation.

Rules:
1. Use ONLY the provided sources - no external knowledge
2. Cite sources using this exact format: [SOURCE:<N>], where <N> is the source number shown below (e.g. [SOURCE:1])
3. If sources lack sufficient information, respond with exactly: REFUSE
4. Keep answers concise (2-4 sentences). Do not repeat the sources.

Question: {question}

Sources:
{sources}

Answer:"""
)


def should_refuse(results: List[Dict], min_score: float = MIN_CONFIDENCE_SCORE) -> tuple:
    if not results:
        return True, "No relevant policies found for this query."
    if results[0]["score"] < min_score:
        return True, f"Insufficient confidence in policy match (score: {results[0]['score']:.2f})."
    return False, None


def format_sources(results: List[Dict]) -> Tuple[str, Dict[int, str]]:
    formatted = []
    index_to_id = {}
    for i, result in enumerate(results, start=1):
        index_to_id[i] = result["chunk_id"]
        formatted.append(f"SOURCE {i}:\n{result['chunk_text']}\n")
    return "\n".join(formatted), index_to_id


def translate_index_citations(answer: str, index_to_id: Dict[int, str]) -> str:
    def _replace(match: "re.Match") -> str:
        chunk_id = index_to_id.get(int(match.group(1)))
        return f"[SOURCE:{chunk_id}]" if chunk_id else match.group(0)
    return SOURCE_INDEX_PATTERN.sub(_replace, answer)


def get_llm(model_name: Optional[str] = None) -> Ollama:
    return Ollama(
        model=model_name or OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.05,
        num_ctx=2048,
    )


def generate_policy_response(
    query: str,
    llm: Optional[Ollama] = None,
    limit: int = 3,
    region: Optional[str] = None,
    content_type: Optional[str] = None,
    policy_source: Optional[str] = None,
    retrieved_results: Optional[List[Dict]] = None,
) -> PolicyResponse:
    start_time = time.time()
    retrieval_ms = 0.0
    model_name = llm.model if llm is not None else OLLAMA_MODEL

    if retrieved_results is not None:
        results = retrieved_results
    else:
        retrieval_start = time.time()
        results = retrieve_policy_chunks(
            query=query,
            limit=limit,
            region=region,
            content_type=content_type,
            policy_source=policy_source,
        )
        retrieval_ms = (time.time() - retrieval_start) * 1000

    refuse, reason = should_refuse(results)
    if refuse:
        return PolicyResponse(
            answer="",
            refused=True,
            refusal_reason=reason,
            latency_ms=(time.time() - start_time) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=0.0,
            model=model_name,
        )

    sources_text, index_to_id = format_sources(results)
    if llm is None:
        llm = get_llm()
        model_name = llm.model

    prompt = POLICY_PROMPT.format(question=query, sources=sources_text)
    try:
        generation_start = time.time()
        raw_answer = llm.invoke(prompt)
        generation_ms = (time.time() - generation_start) * 1000
    except Exception as e:
        return PolicyResponse(
            answer="",
            refused=True,
            refusal_reason=f"LLM generation failed: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=0.0,
            model=model_name,
        )

    raw_answer = THINK_BLOCK_PATTERN.sub("", raw_answer).strip()
    answer = translate_index_citations(raw_answer, index_to_id)
    cited_ids = extract_citations(answer)
    retrieved_ids = {r["chunk_id"] for r in results}
    has_valid_citations = validate_citations(cited_ids, retrieved_ids)

    if not has_valid_citations:
        if REFUSE_TOKEN_PATTERN.search(raw_answer):
            return PolicyResponse(
                answer="",
                refused=True,
                refusal_reason="LLM determined sources insufficient to answer query.",
                latency_ms=(time.time() - start_time) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                model=model_name,
            )
        return PolicyResponse(
            answer="",
            refused=True,
            refusal_reason="Generated response failed citation validation.",
            latency_ms=(time.time() - start_time) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            model=model_name,
        )

    answer = REFUSE_TOKEN_PATTERN.sub("", answer).strip()
    citations = build_citations(cited_ids, results)
    return PolicyResponse(
        answer=answer,
        refused=False,
        citations=citations,
        latency_ms=(time.time() - start_time) * 1000,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        num_tokens_generated=len(answer.split()),
        model=model_name,
    )
