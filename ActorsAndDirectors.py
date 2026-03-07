import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class ActorsAndDirectors:
    """
    LeetCode 30 Days of Pandas – "Actors and Directors Who Cooperated At Least Three Times"
    ----------------------------------------------------------------------------------------
    PROBLEM:
        Given a table `actor_director` with columns:
            actor_id    – ID of the actor
            director_id – ID of the director
            timestamp   – when they worked together (ignored in this solution)

        Return all (actor_id, director_id) pairs that have worked together
        AT LEAST 3 times. Output order does not matter.

    CORE CONCEPT: groupby + agg(count=...) + boolean mask filter
        • groupby(['actor_id', 'director_id'])
              → one bucket per unique pair; a pair that worked together 5 times
                produces ONE bucket with 5 rows in it
        • .agg(count=('director_id', 'count'))
              → named aggregation syntax: count=('source_col', 'func')
              → counts non-null rows in each bucket (i.e. how many collaborations)
              → produces a column called 'count' directly — no rename step needed
        • .reset_index()
              → turns the group keys (actor_id, director_id) back into normal columns
        • stats[stats['count'] >= 3]
              → boolean mask: keep only rows where the pair worked together ≥ 3 times
        • [['actor_id', 'director_id']]
              → column selection: drop the helper 'count' column from the final output

    PANDAS GOTCHAS:
        • Named aggregation syntax  agg(new_name=('col', 'func'))  requires pandas ≥ 0.25
          and is preferred over the dict form because the output column is named in one step
        • 'count' ignores NaN; 'size' counts ALL rows including NaN — here they are equivalent
          because timestamp is always present, but 'count' is the safer default
        • Boolean mask + column selection can be chained in one line; split across two for clarity
    """

    @staticmethod
    def make_sample_data() -> pd.DataFrame:
        """
        Construct actor_director rows that exercise every important case:
          - a pair that hits exactly 3  → must appear in output
          - a pair that exceeds 3 (5x)  → must appear in output
          - a pair with only 2          → must NOT appear
          - a pair with only 1          → must NOT appear
          - same actor, different directors → counted independently
        """
        data = {
            "actor_id": [
                1, 1, 1,          # actor 1 + director 1: exactly 3 times → include
                1, 1, 1, 1, 1,    # actor 1 + director 2: 5 times          → include
                2, 2,             # actor 2 + director 1: only 2 times     → exclude
                3,                # actor 3 + director 3: only 1 time      → exclude
            ],
            "director_id": [
                1, 1, 1,
                2, 2, 2, 2, 2,
                1, 1,
                3,
            ],
            "timestamp": range(11),  # arbitrary; not used by the solution
        }
        df = pd.DataFrame(data)
        log.debug("Sample data created:\n%s\n", df.to_string(index=False))
        return df

    @staticmethod
    def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:

        log.info("=== actors_and_directors ===")
        log.info("Input shape: %d rows x %d cols", *actor_director.shape)

        # ── Step 1: groupby ──────────────────────────────────────────
        # Bucket every row by its (actor, director) pair.
        # No computation happens yet — groupby is lazy.
        grouped_obj = actor_director.groupby(["actor_id", "director_id"])
        log.debug("Unique (actor, director) pairs found: %d", grouped_obj.ngroups)

        # ── Step 2: named aggregation ────────────────────────────────
        # agg(new_col_name=('source_col', 'aggregation_function'))
        # 'count' → non-null row count per bucket = number of collaborations.
        # The output column is named 'count' immediately; no .rename() needed.
        stats = grouped_obj.agg(
            count=("director_id", "count")
        ).reset_index()

        log.debug("After groupby + agg (all pairs with collaboration counts):\n%s\n",
                  stats.to_string(index=False))

        # ── Step 3: filter with boolean mask ─────────────────────────
        # Keep only pairs that collaborated at least 3 times.
        mask = stats["count"] >= 3
        log.debug("Pairs passing the >= 3 threshold: %d / %d",
                  mask.sum(), len(stats))

        # ── Step 4: drop the helper 'count' column ───────────────────
        # The problem only wants actor_id and director_id in the output.
        result = stats[mask][["actor_id", "director_id"]].reset_index(drop=True)

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))

        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        res = result.sort_values(["actor_id", "director_id"]).reset_index(drop=True)

        expected = pd.DataFrame({
            "actor_id":    [1, 1],
            "director_id": [1, 2],
        })

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = ActorsAndDirectors()

    df = solver.make_sample_data()
    result = solver.actors_and_directors(df)
    solver.verify(result)

    # ── Edge case: empty input ───────────────────────────────────────
    log.info("--- Edge case: empty input ---")
    empty = pd.DataFrame(columns=["actor_id", "director_id", "timestamp"])
    empty_result = solver.actors_and_directors(empty)
    log.info("Empty input → %d rows (expected 0)", len(empty_result))

    # ── Edge case: no pair reaches the threshold ─────────────────────
    log.info("--- Edge case: all pairs below threshold ---")
    below = pd.DataFrame({
        "actor_id":    [1, 1, 2],
        "director_id": [1, 2, 1],
        "timestamp":   [0, 1, 2],
    })
    below_result = solver.actors_and_directors(below)
    log.info("Below-threshold input → %d rows (expected 0)", len(below_result))