import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class OrderScores:
    """
    LeetCode 30 Days of Pandas – "Rank Scores"
    -------------------------------------------
    PROBLEM:
        Given a table `scores` with columns:
            id    – unique row identifier
            score – a decimal score value (duplicates possible)

        Return the scores sorted descending with a 'rank' column showing each
        score's rank. Tied scores must receive the SAME rank, and the next
        rank must be consecutive (no gaps). Output columns: score, rank.

    CORE CONCEPT: Series.rank(method='dense', ascending=False)

        pandas rank() assigns a rank to every element in a Series.
        The two parameters here are what make it match the SQL DENSE_RANK():

        method= controls what happens when values TIE:
        ┌───────────┬──────────────────────────────────────────────────────┐
        │ 'average' │ tied values share the average of their would-be ranks│
        │           │ e.g. two values tied for 2nd → both get rank 2.5     │
        │           │ (pandas default)                                     │
        ├───────────┼──────────────────────────────────────────────────────┤
        │ 'min'     │ tied values all get the LOWEST of their would-be ranks│
        │           │ e.g. tied for 2nd/3rd → both get rank 2             │
        │           │ next rank after the tie is 4 (gap!) = SQL RANK()     │
        ├───────────┼──────────────────────────────────────────────────────┤
        │ 'max'     │ tied values all get the HIGHEST would-be rank        │
        ├───────────┼──────────────────────────────────────────────────────┤
        │ 'first'   │ tied values ranked in order of appearance (no ties)  │
        ├───────────┼──────────────────────────────────────────────────────┤
        │ 'dense'   │ tied values share the SAME rank AND the next rank    │
        │           │ is always consecutive — NO GAPS = SQL DENSE_RANK()   │
        └───────────┴──────────────────────────────────────────────────────┘

        ascending=False means rank 1 = highest score (largest value).
        Without this, rank 1 would be the lowest score.

        WHY assign rank before dropping 'id'?
        rank() operates on the score column only, so 'id' being present or
        absent makes no difference — but it is cleaner to compute first,
        then drop and sort in one pass.

        drop('id', axis=1):
        • axis=0 → drop a ROW with that label
        • axis=1 → drop a COLUMN with that name
        Always specify axis explicitly to avoid ambiguity.

    PANDAS GOTCHAS:
        • rank() returns floats by default (e.g. 1.0, 2.0) even for integer-
          compatible ranks. LeetCode accepts this, but .astype(int) removes
          the decimal if you prefer cleaner output.
        • The original DataFrame is mutated by scores['rank'] = ... because
          DataFrames are passed by reference. If you need the input unchanged,
          use scores = scores.copy() at the top of the function.
        • sort_values does not affect the rank values — rank was computed on
          the original order. Sorting is purely for output presentation.
    """

    @staticmethod
    def make_sample_data() -> pd.DataFrame:
        """
        6 rows designed to test every ranking edge case:
          - 3.50 appears twice → both get rank 1, next rank is 2 (not 3)
          - 3.65 appears once  → rank 2
          - 4.00 appears once  → rank 1 (highest score overall... wait — see below)

        Scores in insert order (unsorted) to confirm sort_values is working:
          id=1: 3.50  → rank 2  (tied with id=4)
          id=2: 3.65  → rank 3
          id=3: 4.00  → rank 1  (highest)
          id=4: 3.50  → rank 2  (tied with id=1)
          id=5: 3.45  → rank 5  (note: 3.40 is rank 6, so there is a gap... no,
                                  dense fills it → rank 4 after 3.65 and 3.50)
          id=6: 3.40  → rank 5  (lowest, dense rank)

        Dense rank ordering (descending):
          4.00 → 1
          3.65 → 2
          3.50 → 3  (two rows, same rank)
          3.45 → 4
          3.40 → 5
        """
        return pd.DataFrame({
            "id":    [1,    2,    3,    4,    5,    6   ],
            "score": [3.50, 3.65, 4.00, 3.50, 3.45, 3.40],
        })

    @staticmethod
    def order_scores(scores: pd.DataFrame) -> pd.DataFrame:

        log.info("=== order_scores ===")
        log.info("Input shape: %d rows x %d cols", *scores.shape)
        log.debug("Input:\n%s\n", scores.to_string(index=False))

        # ── Step 1: compute dense rank descending ────────────────────
        # method='dense' → no gaps between ranks after ties (SQL DENSE_RANK)
        # ascending=False → rank 1 = highest score
        scores = scores.copy()  # avoid mutating the caller's DataFrame
        scores["rank"] = scores["score"].rank(method="dense", ascending=False)

        log.debug("After rank() — all columns including id:\n%s\n",
                  scores.to_string(index=False))

        # ── Step 2: show all ranking methods side by side ────────────
        for method in ("average", "min", "max", "first", "dense"):
            scores[f"rank_{method}"] = scores["score"].rank(
                method=method, ascending=False
            )
        log.debug("All rank methods compared:\n%s\n",
                  scores[["score", "rank_average", "rank_min",
                           "rank_max", "rank_first", "rank_dense"]]
                  .sort_values("score", ascending=False)
                  .to_string(index=False))
        # Drop the comparison columns before returning
        scores = scores[["id", "score", "rank"]]

        # ── Step 3: drop 'id', sort by score descending ──────────────
        # axis=1 → drop a column (axis=0 would drop a row)
        result = (
            scores
            .drop("id", axis=1)
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
        )

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))
        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        assert list(result.columns) == ["score", "rank"], \
            f"Wrong columns: {list(result.columns)}"

        expected = pd.DataFrame({
            "score": [4.00, 3.65, 3.50, 3.50, 3.45, 3.40],
            "rank":  [1.0,  2.0,  3.0,  3.0,  4.0,  5.0 ],
        })

        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = OrderScores()

    df = solver.make_sample_data()
    result = solver.order_scores(df)
    solver.verify(result)

    # ── Demonstrate the gap difference: 'min' vs 'dense' ─────────────
    log.info("--- Contrast: min (SQL RANK) vs dense (SQL DENSE_RANK) ---")
    sample = pd.Series([100, 90, 90, 80])
    log.info("Values:        %s", sample.tolist())
    log.info("rank min:      %s  ← gap after tie (1, 2, 2, 4)",
             sample.rank(method="min", ascending=False).tolist())
    log.info("rank dense:    %s  ← no gap after tie (1, 2, 2, 3)",
             sample.rank(method="dense", ascending=False).tolist())

    # ── Edge case: all scores identical ──────────────────────────────
    log.info("--- Edge case: all scores the same ---")
    all_same = pd.DataFrame({"id": [1, 2, 3], "score": [5.0, 5.0, 5.0]})
    result2 = solver.order_scores(all_same)
    log.info("All tied → all rank 1:\n%s\n", result2.to_string(index=False))

    # ── Edge case: single row ─────────────────────────────────────────
    log.info("--- Edge case: single row ---")
    single = pd.DataFrame({"id": [1], "score": [9.99]})
    result3 = solver.order_scores(single)
    log.info("Single row → rank 1:\n%s\n", result3.to_string(index=False))