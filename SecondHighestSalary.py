import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class SecondHighestSalary:
    """
    LeetCode 30 Days of Pandas – "Second Highest Salary"
    -----------------------------------------------------
    PROBLEM:
        Given a table `employee` with columns:
            id     – employee's unique ID
            salary – employee's salary (duplicates possible)

        Return a single-cell DataFrame with column 'SecondHighestSalary'
        containing the second highest DISTINCT salary.
        If no second distinct salary exists, return NULL (None).

    CORE CONCEPT: drop_duplicates → nlargest → iloc → ternary None guard

        STEP 1 — drop_duplicates()
            employee.salary.drop_duplicates()
            Removes repeated salary values so that two employees on the
            same salary don't both count as "distinct" salaries.
            e.g. [900, 900, 800] → [900, 800] → second highest is 800, not 900.

        STEP 2 — nlargest(2)
            s.nlargest(2)
            Returns the 2 largest values from the deduplicated Series,
            sorted descending. Does NOT reset the index — original positions
            are preserved.

            nlargest(n) vs sort_values().head(n):
            ┌────────────────────┬──────────────────────────────────────────┐
            │ nlargest(n)        │ efficient partial sort; O(k log n) where │
            │                    │ k=len(series); returns n largest values  │
            ├────────────────────┼──────────────────────────────────────────┤
            │ sort_values().head │ full sort O(k log k); equivalent result  │
            │                    │ but wasteful when you only need top-n    │
            └────────────────────┴──────────────────────────────────────────┘
            For LeetCode scale, either works; nlargest is idiomatic.

        STEP 3 — iloc[-1]
            .iloc[-1] selects the LAST element of the nlargest result.
            After nlargest(2), the Series has [highest, second_highest].
            iloc[-1] reliably picks the second-highest without hardcoding
            an integer index (which would vary because drop_duplicates
            preserves original index positions).

            iloc vs loc:
            • iloc — position-based  (iloc[0] = first row regardless of index)
            • loc  — label-based     (loc[0] = row whose index label is 0)
            Always use iloc when you want positional access after operations
            that may scramble the index labels.

        STEP 4 — ternary None guard + pd.DataFrame wrapper
            s.nlargest(2).iloc[-1] if len(s) > 1 else None
            If there is only one distinct salary (or none), nlargest(2) would
            return a Series of length 1 and iloc[-1] would still return the
            only element — giving the HIGHEST salary instead of None.
            The guard `len(s) > 1` catches this and returns None explicitly.

            pd.DataFrame({'SecondHighestSalary': [value]})
            Wrapping in a list [ ] is required — passing a scalar directly
            raises a ValueError because DataFrame needs an iterable.

    PANDAS GOTCHAS:
        • drop_duplicates() preserves the FIRST occurrence and its original
          index label. After nlargest, iloc[-1] is still safe because iloc
          works on position, not label.
        • None in a numeric column becomes NaN (float) when stored in a
          DataFrame — that is the expected pandas representation of SQL NULL.
        • nlargest(2) on a Series of length 1 returns length-1, not length-2;
          it does not pad with NaN. This is why the length guard is essential.
    """

    @staticmethod
    def make_sample_data() -> dict[str, pd.DataFrame]:
        """
        Four DataFrames covering every important case:

          'normal'    – multiple distinct salaries → clear second highest
          'dupes'     – duplicated top salary → second is still the next distinct value
          'one_row'   – single employee → no second salary → None
          'all_same'  – all salaries identical → only one distinct value → None
        """
        samples = {
            "normal": pd.DataFrame({
                "id":     [1,   2,   3  ],
                "salary": [300, 200, 100],
            }),
            "dupes": pd.DataFrame({
                "id":     [1,   2,   3,   4  ],
                "salary": [300, 300, 200, 100],  # 300 appears twice → second highest is 200
            }),
            "one_row": pd.DataFrame({
                "id":     [1  ],
                "salary": [100],                 # only one row → None
            }),
            "all_same": pd.DataFrame({
                "id":     [1,   2,   3  ],
                "salary": [500, 500, 500],        # one distinct value → None
            }),
        }
        for label, df in samples.items():
            log.debug("Sample '%s':\n%s\n", label, df.to_string(index=False))
        return samples

    @staticmethod
    def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:

        log.info("=== second_highest_salary ===")
        log.info("Input shape: %d rows x %d cols", *employee.shape)

        # ── Step 1: deduplicate salaries ─────────────────────────────
        # We want the second highest DISTINCT salary, so collapse repeats first.
        s = employee["salary"].drop_duplicates()

        log.debug("Salaries after drop_duplicates: %s", s.tolist())
        log.debug("Distinct salary count: %d", len(s))

        # ── Step 2 & 3: get the two largest, pick the last ───────────
        # Guard: if fewer than 2 distinct salaries exist, return None.
        # Without the guard, nlargest(2).iloc[-1] on a length-1 Series
        # would silently return the highest salary instead of None.
        if len(s) > 1:
            top2 = s.nlargest(2)
            log.debug("nlargest(2) result (index preserved): %s",
                      top2.to_dict())
            second = top2.iloc[-1]
            log.debug("Second highest salary (iloc[-1]): %s", second)
        else:
            second = None
            log.debug("Fewer than 2 distinct salaries → returning None")

        # ── Step 4: wrap in a single-row DataFrame ───────────────────
        # [second] in a list is required; passing a bare scalar raises ValueError.
        result = pd.DataFrame({"SecondHighestSalary": [second]})

        log.info("Result:\n%s\n", result.to_string(index=False))
        return result

    @staticmethod
    def verify(result: pd.DataFrame, expected_value) -> None:
        assert list(result.columns) == ["SecondHighestSalary"], \
            f"Wrong column name: {list(result.columns)}"
        assert len(result) == 1, \
            f"Expected exactly 1 row, got {len(result)}"

        actual = result["SecondHighestSalary"].iloc[0]

        if expected_value is None:
            assert pd.isna(actual), f"Expected None/NaN, got {actual}"
        else:
            assert actual == expected_value, \
                f"Expected {expected_value}, got {actual}"

        log.info("✅  Assertion passed — SecondHighestSalary = %s", actual)


if __name__ == "__main__":
    solver = SecondHighestSalary()
    samples = solver.make_sample_data()

    cases = [
        ("normal",   200),
        ("dupes",    200),   # 300 is duplicated but still only the highest distinct
        ("one_row",  None),
        ("all_same", None),
    ]

    for label, expected in cases:
        log.info("=== Case: '%s' (expected: %s) ===", label, expected)
        result = solver.second_highest_salary(samples[label])
        solver.verify(result, expected)

    # ── Show why the None guard is essential ─────────────────────────
    log.info("--- Demonstrating why len(s) > 1 guard is essential ---")
    one_salary = pd.Series([500])
    log.info("nlargest(2) on a 1-element Series: %s", one_salary.nlargest(2).tolist())
    log.info("iloc[-1] of that result: %s  ← would be returned as 'second highest' without guard",
             one_salary.nlargest(2).iloc[-1])

    # ── Show iloc vs loc pitfall ──────────────────────────────────────
    log.info("--- iloc vs loc after drop_duplicates ---")
    messy = pd.Series([300, 300, 200, 100], index=[10, 11, 12, 13])
    deduped = messy.drop_duplicates()
    top2 = deduped.nlargest(2)
    log.info("Original index labels preserved after nlargest: %s", top2.to_dict())
    log.info("iloc[-1] = %s  (position-based, always correct)", top2.iloc[-1])
    log.info("loc[-1]  would raise KeyError because -1 is not an index label")