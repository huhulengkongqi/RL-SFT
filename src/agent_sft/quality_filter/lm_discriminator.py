"""LLM-based quality discriminator for evolved prompts."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LMDiscriminatorResult:
    """Result from LLM discriminator judgment."""

    def __init__(self, is_valid: bool, reason: str, score: float, raw_response: str):
        self.is_valid = is_valid
        self.reason = reason
        self.score = score
        self.raw_response = raw_response


class LMDiscriminator:
    """
    Uses LLM to judge the quality of evolved prompts.

    Filters out:
    1. No information gain (trivial changes only)
    2. Stopwords only (meaningless prompt)
    3. Exact copy (or near copy) of original
    4. Unsolvable / nonsensical prompts
    """

    SYSTEM_PROMPT = """你是一个prompt质量判别器。请判断以下进化后的prompt质量是否合格。

评判标准：
1. ❌ 【无信息增益】：只是简单改写、重新措辞，没有实质变化
2. ❌ 【仅含停用词】：prompt没有实质内容，只有礼貌用语或无意义词汇
3. ❌ 【复制原始】：几乎完全复制原prompt，没有任何进化
4. ❌ 【无法执行】：prompt描述模糊、矛盾、或无法理解的任务
5. ✅ 【合格】：prompt有实质变化，增加了难度或多样性，是有意义的任务

请以JSON格式输出：
{
    "is_valid": true/false,
    "reason": "判断理由",
    "score": 0到1之间的质量分数
}

质量分数说明：
- 0.0-0.3: 完全不合格
- 0.4-0.6: 勉强合格但质量较低
- 0.7-0.8: 合格
- 0.9-1.0: 优质进化，质量很高"""

    def __init__(
        self,
        llm_client: Any,
        max_concurrent_requests: int = 5,
        temperature: float = 0.3,
        retry_attempts: int = 3,
    ):
        """
        Initialize LLM discriminator.

        Args:
            llm_client: LLM client with `achat()` method
        """
        self.llm_client = llm_client
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._retry_strategy = AsyncRetrying(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(min=1.0, max=10.0),
        )

        # Simple stopwords for quick filtering (before LLM call)
        self._stopwords_chinese = {
            "请", "帮我", "你好", "谢谢", "的", "是", "在", "有", "和", "与",
            "问题", "任务", "解决", "处理", "计算", "分析",
        }
        self._stopwords_english = {
            "please", "help", "hello", "thanks", "the", "is", "a", "an", "and",
            "problem", "task", "solve", "compute", "calculate", "analyze",
        }

    def _quick_heuristic_check(self, prompt: str, original_prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Quick heuristic checks before calling LLM.

        Returns:
            (is_valid, reason) - reason is None if passed, or failure reason
        """
        # Check exact copy
        if prompt.strip() == original_prompt.strip():
            return False, "Exact copy of original prompt"

        # Check jaccard similarity (too similar)
        prompt_words = set(prompt.lower().split())
        original_words = set(original_prompt.lower().split())
        if prompt_words and original_words:
            intersection = len(prompt_words & original_words)
            union = len(prompt_words | original_words)
            jaccard = intersection / union if union > 0 else 1.0
            if jaccard > 0.95:
                return False, f"Near copy of original (jaccard={jaccard:.3f})"

        # Check if mostly stopwords
        words = prompt.lower().split()
        if len(words) < 5:
            return False, "Prompt too short"

        stopword_count = sum(1 for w in words if w.strip(".,!?") in self._stopwords_chinese)
        stopword_count += sum(1 for w in words if w.strip(".,!?") in self._stopwords_english)
        stopword_ratio = stopword_count / len(words)

        if stopword_ratio > 0.8:
            return False, f"Mostly stopwords"

        return True, None

    async def _judge_single(self, prompt: str, original_prompt: str) -> LMDiscriminatorResult:
        """
        Judge a single evolved prompt.

        Args:
            prompt: Evolved prompt text
            original_prompt: Original seed prompt text

        Returns:
            Discriminator result
        """
        # First run quick heuristic checks
        heuristic_valid, heuristic_reason = self._quick_heuristic_check(prompt, original_prompt)
        if not heuristic_valid:
            return LMDiscriminatorResult(
                is_valid=False, reason=heuristic_reason or "Heuristic check", score=0.0, raw_response="")

        user_prompt = f"""原prompt：
{original_prompt}

进化后的prompt：
{prompt}

请判断进化后的prompt质量。"""

        try:
            async with self._semaphore:
                async for attempt in self._retry_strategy:
                    with attempt:
                        model_name = getattr(self.llm_client, "model", "default")
                        response = await self.llm_client.achat(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": self.SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=self.temperature,
                            max_tokens=512,
                        )

                        # Try to parse JSON from response
                        json_match = re.search(r"\{.*\}", response, re.DOTALL)
                        if json_match:
                            try:
                                result_json = json.loads(json_match.group())
                                return LMDiscriminatorResult(
                                    is_valid=bool(result_json.get("is_valid", True)),
                                    reason=str(result_json.get("reason", "No reason provided")),
                                    score=float(result_json.get("score", 0.5)),
                                    raw_response=response,
                                )
                            except json.JSONDecodeError:
                                pass

                        # Fallback: assume valid if LLM didn't fail
                        return LMDiscriminatorResult(
                            is_valid=True,
                            reason="JSON parse failed, default to valid",
                            score=0.5,
                            raw_response=response,
                        )
        except Exception as e:
            logger.warning(f"Discriminator failed: {e}")
            return LMDiscriminatorResult(
                is_valid=True, reason=f"Discriminator error: {e}", score=0.6, raw_response=str(e)
            )

    async def judge_batch(
        self,
        prompts: List[Dict],
        original_prompts: Dict[str, str],
        prompt_key: str = "prompt",
        id_key: str = "id",
        parent_id_key: str = "parent_id",
    ) -> List[LMDiscriminatorResult]:
        """
        Judge a batch of evolved prompts.

        Args:
            prompts: List of evolved prompt dicts
            original_prompts: Dict mapping prompt id to original prompt text
            prompt_key: Key for prompt text
            id_key: Key for prompt id
            parent_id_key: Key for parent prompt id

        Returns:
            List of discriminator results
        """
        tasks = []
        for prompt in prompts:
            parent_id = prompt.get("evolution_metadata", {}).get(parent_id_key, "")
            original = original_prompts.get(parent_id, "")
            if not original:
                original = prompt.get("original_prompt", "")

            tasks.append(self._judge_single(prompt[prompt_key], original))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in results:
            if isinstance(result, LMDiscriminatorResult):
                processed.append(result)
            else:
                processed.append(
                    LMDiscriminatorResult(
                        is_valid=True,
                        reason=f"Exception: {str(result)}",
                        score=0.5,
                        raw_response=str(result),
                    )
                )

        return processed

    async def filter_prompts(
        self,
        prompts: List[Dict],
        original_prompts: Dict[str, str],
        min_score: float = 0.5,
    ) -> List[Dict]:
        """
        Filter prompts using discriminator.

        Args:
            prompts: List of prompt dicts
            original_prompts: Dict mapping parent id to original prompt text
            min_score: Minimum quality score to keep

        Returns:
            Filtered list of prompts with discriminator metadata added
        """
        results = await self.judge_batch(prompts, original_prompts)

        filtered = []
        for prompt, result in zip(prompts, results):
            if result.is_valid and result.score >= min_score:
                prompt["evolution_metadata"]["discriminator_score"] = result.score
                prompt["evolution_metadata"]["discriminator_reason"] = result.reason
                filtered.append(prompt)

        logger.info(f"LM discriminator: {len(prompts)} → {len(filtered)} (filtered {len(prompts) - len(filtered)})")
        return filtered
