"""End-to-end evolution pipeline."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evolver import EvolInstruct, EvolInstructConfig
from .models import EvolvedPrompt, GenerationStats

from ..quality_filter.lm_discriminator import LMDiscriminator
from ..quality_filter.metrics import DiversityMetrics

# Simple hash-based deduplicator
class HashDeduplicator:
    """Simple deduplicator using MD5 hashes to avoid sentence-transformers dependency issues."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        logger.info("Using HashDeduplicator (sentence-transformers disabled for stability)")

    def deduplicate(self, prompts: List[Dict], prompt_key: str = "prompt",
                   quality_key: str = "quality_score", id_key: str = "id") -> List[Dict]:
        if len(prompts) <= 1:
            return prompts

        seen_hashes = set()
        kept_prompts = []
        sorted_prompts = sorted(prompts, key=lambda x: x.get(quality_key, 0.0), reverse=True)

        for prompt in sorted_prompts:
            content_hash = hashlib.md5(prompt[prompt_key].encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                kept_prompts.append(prompt)

        logger.info(f"Hash deduplication: {len(prompts)} → {len(kept_prompts)}")
        return kept_prompts

logger = logging.getLogger(__name__)


class EvolutionPipeline:
    """
    End-to-end evolution pipeline:
    1. Evolve prompts using EvolInstruct
    2. Deduplicate using embeddings
    3. Filter using LLM discriminator
    4. Track statistics
    5. Save results
    """

    def __init__(
        self,
        llm_client: Any,
        evolver_config: Optional[EvolInstructConfig] = None,
        similarity_threshold: float = 0.85,
        discriminator_min_score: float = 0.5,
        use_hash_dedup: bool = True,
    ):
        """
        Initialize evolution pipeline.

        Args:
            llm_client: LLM client for evolution and discrimination
            evolver_config: Configuration for the evolver
            similarity_threshold: Embedding similarity threshold for deduplication
            discriminator_min_score: Minimum score from discriminator to keep prompt
            use_hash_dedup: Use hash-based deduplication instead of embeddings (faster, more stable)
        """
        self.evolver = EvolInstruct(llm_client, evolver_config)
        self.deduplicator = HashDeduplicator(similarity_threshold=similarity_threshold)
        self.discriminator = LMDiscriminator(llm_client)
        self.metrics = DiversityMetrics()
        self.discriminator_min_score = discriminator_min_score

        self.stats_history: List[GenerationStats] = []
        self.all_prompts: List[Dict] = []
        self.original_prompts: Dict[str, str] = {}

    def load_seeds(self, seed_file: str) -> List[Dict]:
        """Load seed prompts from JSON file."""
        with open(seed_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both formats: direct array or {"prompts": [...]}
        if isinstance(data, dict) and "prompts" in data:
            seeds = data["prompts"]
        else:
            seeds = data

        self.original_prompts = {s["id"]: s["prompt"] for s in seeds}
        logger.info(f"Loaded {len(seeds)} seed prompts from {seed_file}")
        return seeds

    async def run_single_generation(
        self,
        seeds: List[Dict],
        generation: int,
        num_evolutions_per_seed: int = 3,
    ) -> List[Dict]:
        """
        Run a single generation of evolution.

        Args:
            seeds: Seed prompts for this generation
            generation: Generation number
            num_evolutions_per_seed: How many evolved variants per seed

        Returns:
            List of valid evolved prompts
        """
        initial_count = len(seeds)
        logger.info(f"=== Generation {generation}: Starting with {initial_count} seeds ===")

        # Step 1: Evolve - create multiple variants per seed
        seeds_to_evolve = []
        for seed in seeds:
            for _ in range(num_evolutions_per_seed):
                seeds_to_evolve.append(seed.copy())

        logger.info(f"Step 1: Evolving {len(seeds_to_evolve)} variants...")
        evolution_results = await self.evolver.evolve_batch(seeds_to_evolve, generation=generation)

        evolved_prompts = []
        for result in evolution_results:
            if result.success and result.evolved_prompt:
                evolved_prompts.append(result.evolved_prompt.dict())

        evolved_count = len(evolved_prompts)
        strategy_stats = self.evolver.get_strategy_stats(evolution_results)
        logger.info(f"Step 1: Successfully evolved {evolved_count} prompts")
        logger.info(f"Strategy distribution: {dict(strategy_stats)}")

        if not evolved_prompts:
            logger.warning("No prompts evolved!")
            return []

        # Step 2: Embedding deduplication
        logger.info("Step 2: Embedding deduplication...")
        deduped_prompts = self.deduplicator.deduplicate(evolved_prompts)
        after_dedup_count = len(deduped_prompts)
        dedup_filter_rate = 1.0 - (after_dedup_count / evolved_count) if evolved_count > 0 else 0.0
        logger.info(f"Step 2: After deduplication: {after_dedup_count} (filter rate: {dedup_filter_rate:.2%})")

        # Step 3: LLM discriminator filtering
        if self.discriminator_min_score > 0:
            logger.info(f"Step 3: LLM quality discrimination (min_score={self.discriminator_min_score})...")
            valid_prompts = await self.discriminator.filter_prompts(
                deduped_prompts,
                original_prompts=self.original_prompts,
                min_score=self.discriminator_min_score,
            )
        else:
            logger.info("Step 3: Skipping LLM quality discrimination (min_score=0)")
            valid_prompts = deduped_prompts

        final_count = len(valid_prompts)
        disc_filter_rate = 1.0 - (final_count / after_dedup_count) if after_dedup_count > 0 else 0.0
        total_filter_rate = 1.0 - (final_count / evolved_count) if evolved_count > 0 else 0.0
        logger.info(f"Step 3: After quality filter: {final_count} (filter rate: {disc_filter_rate:.2%})")

        # Step 4: Calculate diversity metrics
        logger.info("Step 4: Calculating diversity metrics...")
        all_prompts_text = [p["prompt"] for p in valid_prompts]
        self_bleu = self.metrics.self_bleu(all_prompts_text) if all_prompts_text else 0.0

        # Record statistics
        stats = GenerationStats(
            generation=generation,
            initial_count=initial_count,
            evolved_count=evolved_count,
            after_deduplication_count=after_dedup_count,
            final_valid_count=final_count,
            deduplication_filter_rate=dedup_filter_rate,
            discriminator_filter_rate=disc_filter_rate,
            total_filter_rate=total_filter_rate,
            self_bleu=self_bleu,
            strategies_used=strategy_stats,
        )
        self.stats_history.append(stats)
        logger.info(f"Generation {generation} complete: {stats}")

        return valid_prompts

    async def run(
        self,
        seed_file: str,
        num_generations: int = 4,
        evolutions_per_seed: int = 3,
        output_dir: str = "data/evolved",
    ) -> Dict[str, Any]:
        """
        Run full evolution pipeline for multiple generations.

        Args:
            seed_file: Path to seed prompts JSON
            num_generations: Number of evolution rounds to run
            evolutions_per_seed: Number of variants to create per seed
            output_dir: Directory to save results

        Returns:
            Final results with all prompts and statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load initial seeds
        seeds = self.load_seeds(seed_file)
        current_seeds = seeds.copy()

        # Add seeds to original prompts mapping
        for seed in seeds:
            if seed["id"] not in self.original_prompts:
                self.original_prompts[seed["id"]] = seed["prompt"]

        self.all_prompts.extend(seeds)

        # Run evolution for each generation
        for gen in range(1, num_generations + 1):
            evolved = await self.run_single_generation(
                current_seeds,
                generation=gen,
                num_evolutions_per_seed=evolutions_per_seed,
            )

            if not evolved:
                logger.warning(f"Generation {gen} produced no prompts. Stopping early.")
                break

            self.all_prompts.extend(evolved)

            # Use evolved prompts as seeds for next generation (mix with some originals)
            import random

            original_sample = random.sample(seeds, min(50, len(seeds)))
            current_seeds = evolved + original_sample

            # Save intermediate results
            intermediate_file = output_path / f"generation_{gen}.json"
            gen_prompts = [p for p in self.all_prompts if "evolution_metadata" in p or p.get("source") == "evolved"]
            for p in gen_prompts:
                if "created_at" in p and hasattr(p["created_at"], "isoformat"):
                    p["created_at"] = p["created_at"].isoformat()
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(gen_prompts, f, indent=2, ensure_ascii=False)

        # Save final results
        final_file = output_path / "final_evolved.json"
        evolved_only = [p for p in self.all_prompts if "evolution_metadata" in p or p.get("source") == "evolved"]
        # Convert datetime values for JSON serialization
        for p in evolved_only:
            if "created_at" in p and hasattr(p["created_at"], "isoformat"):
                p["created_at"] = p["created_at"].isoformat()
            if "evolution_metadata" in p and "timestamp" in p["evolution_metadata"]:
                if hasattr(p["evolution_metadata"]["timestamp"], "isoformat"):
                    p["evolution_metadata"]["timestamp"] = p["evolution_metadata"]["timestamp"].isoformat()
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(evolved_only, f, indent=2, ensure_ascii=False)

        # Save statistics
        stats_file = output_path / "evolution_stats.json"
        stats_dict = {
            "summary": {
                "total_evolved": len(evolved_only),
                "total_seeds": len(seeds),
                "num_generations": num_generations,
                "final_self_bleu": self.stats_history[-1].self_bleu if self.stats_history else 0.0,
            },
            "generations": [],
        }
        for stat in self.stats_history:
            stat_dict = stat.dict()
            # Convert datetime and enum values to strings
            for k, v in stat_dict.items():
                if isinstance(v, dict):
                    stat_dict[k] = {str(kk): vv for kk, vv in v.items()}
                elif hasattr(v, "isoformat"):
                    stat_dict[k] = v.isoformat()
            stats_dict["generations"].append(stat_dict)

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"=== Evolution Complete ===")
        logger.info(f"Total evolved prompts: {len(evolved_only)}")
        logger.info(f"Results saved to: {output_dir}")

        # Print summary table
        print("\n" + "=" * 100)
        print("EVOLUTION SUMMARY")
        print("=" * 100)
        print(f"{'Gen':<5} {'Initial':<10} {'Evolved':<10} {'Deduped':<10} {'Valid':<10} "
              f"{'Filter%':<10} {'Self-BLEU':<12}")
        print("-" * 100)
        for stat in self.stats_history:
            print(f"{stat.generation:<5} {stat.initial_count:<10} {stat.evolved_count:<10} "
                  f"{stat.after_deduplication_count:<10} {stat.final_valid_count:<10} "
                  f"{stat.total_filter_rate:<10.2%} {stat.self_bleu:<12.4f}")
        print("=" * 100)
        print(f"Total evolved prompts: {len(evolved_only)}")
        print(f"Output directory: {output_dir}")

        return stats_dict


async def run_evolution_pipeline(
    seed_file: str,
    llm_client: Any,
    num_generations: int = 4,
    evolutions_per_seed: int = 3,
    output_dir: str = "data/evolved",
) -> Dict[str, Any]:
    """Convenience function to run evolution pipeline."""
    pipeline = EvolutionPipeline(llm_client)
    return await pipeline.run(
        seed_file=seed_file,
        num_generations=num_generations,
        evolutions_per_seed=evolutions_per_seed,
        output_dir=output_dir,
    )
