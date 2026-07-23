import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

from app.retrieval import retrieve_policy_chunks
from app.schemas import PolicyResponse
from app.citations import extract_citations, validate_citations, build_citations

MIN_CONFIDENCE_SCORE = 0.25
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Smaller models reliably hallucinate a few characters of a 36-char UUID when asked
# to copy it verbatim. Sources are labeled with short numeric indices in the prompt
# instead, and the model's [SOURCE:<N>] citations get translated back to real chunk_ids
# post-generation -- see format_sources() / translate_index_citations().
SOURCE_INDEX_PATTERN = re.compile(r"\[SOURCE:(\d+)\]")
REFUSE_TOKEN_PATTERN = re.compile(r"(?<!\w)REFUSE(?!\w)[.:,]?\s*")

POLICY_PROMPT = PromptTemplate(
    input_variables=["question", "sources"],
    template="""You are a policy compliance assistant for Google Ads.

Answer using ONLY the sources below. Every factual claim MUST include a citation.

Rules:
1. Use ONLY the provided sources - no external knowledge
2. Cite sources using this exact format: [SOURCE:<N>], where <N> is the source number shown below (e.g. [SOURCE:1])
3. If sources lack sufficient information, respond with exactly: REFUSE

Question: {question}

Sources:
{sources}

Answer:"""
)


def should_refuse(results: List[Dict], min_score: float = MIN_CONFIDENCE_SCORE) -> tuple[bool, Optional[str]]:
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
    """Replaces [SOURCE:<N>] index citations with the real [SOURCE:<chunk_id>] they refer to."""
    def _replace(match: "re.Match") -> str:
        chunk_id = index_to_id.get(int(match.group(1)))
        return f"[SOURCE:{chunk_id}]" if chunk_id else match.group(0)

    return SOURCE_INDEX_PATTERN.sub(_replace, answer)


def get_llm(model_name: Optional[str] = None) -> Ollama:
    return Ollama(
        model=model_name or OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.05,
        num_ctx=3072,
    )


def generate_policy_response(
    query: str,
    llm: Optional[Ollama] = None,
    limit: int = 3,
    region: Optional[str] = None,
    content_type: Optional[str] = None,
    policy_source: Optional[str] = None,
    retrieved_results: Optional[List[Dict]] = None
) -> PolicyResponse:
    start_time = time.time()

    results = retrieved_results if retrieved_results is not None else retrieve_policy_chunks(
        query=query,
        limit=limit,
        region=region,
        content_type=content_type,
        policy_source=policy_source
    )
    
    refuse, reason = should_refuse(results)
    if refuse:
        latency_ms = (time.time() - start_time) * 1000
        return PolicyResponse(
            answer="",
            refused=True,
            refusal_reason=reason,
            latency_ms=latency_ms
        )
    
    sources_text, index_to_id = format_sources(results)

    if llm is None:
        llm = get_llm()

    chain = LLMChain(llm=llm, prompt=POLICY_PROMPT)

    try:
        generation_start = time.time()
        raw_answer = chain.run(question=query, sources=sources_text)
        generation_time = (time.time() - generation_start) * 1000
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return PolicyResponse(
            answer="",
            refused=True,
            refusal_reason=f"LLM generation failed: {str(e)}",
            latency_ms=latency_ms
        )

    answer = translate_index_citations(raw_answer, index_to_id)
    cited_ids = extract_citations(answer)
    retrieved_ids = {r["chunk_id"] for r in results}
    has_valid_citations = validate_citations(cited_ids, retrieved_ids)

    # Smaller models often tack a stray "REFUSE" onto an otherwise valid, cited answer
    # instead of using it as a clean binary signal. Only treat it as a real refusal when
    # there's no valid cited content to fall back on.
    if not has_valid_citations:
        if REFUSE_TOKEN_PATTERN.search(raw_answer):
            latency_ms = (time.time() - start_time) * 1000
            return PolicyResponse(
                answer="",
                refused=True,
                refusal_reason="LLM determined sources insufficient to answer query.",
                latency_ms=latency_ms
            )

        latency_ms = (time.time() - start_time) * 1000
        return PolicyResponse(
            answer="",
            refused=True,
            refusal_reason="Generated response failed citation validation.",
            latency_ms=latency_ms
        )

    answer = REFUSE_TOKEN_PATTERN.sub("", answer).strip()
    citations = build_citations(cited_ids, results)
    
    num_tokens = len(answer.split())
    latency_ms = (time.time() - start_time) * 1000
    
    return PolicyResponse(
        answer=answer,
        refused=False,
        citations=citations,
        latency_ms=latency_ms,
        num_tokens_generated=num_tokens
    )


if __name__ == "__main__":
    print("Testing Generation Pipeline")
    print()
    
    test_queries = [
        "Can I advertise alcohol?",
        "What are the requirements for advertising healthcare products?",
        "Can I use trademarked terms in my ad copy?",
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"Test {i}/3: {test_query}")
        
        response = generate_policy_response(test_query, limit=5)
        
        print(f"Refused: {response.refused}")
        
        if response.refused:
            print(f"Reason: {response.refusal_reason}")
        else:
            print(f"\nAnswer: {response.answer}\n")
            print(f"Citations: {len(response.citations)}")
            for j, citation in enumerate(response.citations, 1):
                print(f"  {j}. {citation.policy_path}")
        
        if response.latency_ms:
            print(f"Latency: {response.latency_ms:.1f}ms")
        if response.num_tokens_generated:
            print(f"Tokens: {response.num_tokens_generated}")
        
        print()
    
    print("Testing complete")
