import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class CountSalaryCategories:
    """
    LeetCode 30 Days of Pandas – "Count Salary Categories"
    -------------------------------------------------------
    PROBLEM:
        Given a table `accounts` with columns:
            account_id – unique account identifier
            income     – the account's income

        Categorise every account into exactly one bucket:
            'Low Salary'     – income < 20,000
            'Average Salary' – 20,000 <= income <= 50,000
            'High Salary'    – income > 50,000

        Return ALL THREE category rows even if a category has zero accounts.
        Output columns: category, accounts_count.

    CORE CONCEPT: manual DataFrame construction with per-category boolean filters

        Rather than groupby (which would silently omit empty categories),
        this solution hardcodes the three rows and computes each count
        independently using boolean masks + .shape[0].

        .shape[0] vs .sum() vs len():
        ┌─────────────────────────────┬──────────────────────────────────────┐
        │ df[mask].shape[0]           │ number of rows passing the mask;     │
        │                             │ .shape returns (rows, cols) tuple    │
        ├─────────────────────────────┼──────────────────────────────────────┤
        │ mask.sum()                  │ counts True values in a boolean      │
        │                             │ Series — equivalent and more concise │
        ├─────────────────────────────┼──────────────────────────────────────┤
        │ len(df[mask])               │ also equivalent; least idiomatic     │
        └─────────────────────────────┴──────────────────────────────────────┘
        All three produce the same integer. mask.sum() is preferred when you
        already have a boolean Series — it avoids creating a filtered copy.

        WHY NOT groupby + pd.cut?
            pd.cut(accounts.income, bins=[-inf, 19999, 50000, inf]) + groupby
            would be more "pandas-idiomatic" for bucketing, BUT it would return
            only the categories that actually have data. You would then need to
            reindex against all three category labels and fill missing counts
            with 0. The explicit approach here is simpler and more readable for
            this specific problem.

        COMPOUND BOOLEAN MASK:
            accounts[(accounts.income >= 20000) & (accounts.income <= 50000)]
            • Each condition in parentheses produces a boolean Series
            • & is element-wise AND for Series (not Python's 'and' keyword)
            • 'and' / 'or' raise ValueError on Series — always use & / |
            • Parentheses around each condition are REQUIRED because & has
              higher operator precedence than >= and <=

    PANDAS GOTCHAS:
        • The output category order is fixed by the list — pandas preserves
          insertion order in DataFrame construction from a dict.
        • If accounts is empty, all three counts are 0 — the DataFrame is
          still returned with all three rows (correct behaviour).
        • income exactly at 20,000 or 50,000 must land in 'Average Salary'.
          The >= and <= operators handle this correctly; using > 19999 as an
          integer shortcut would break on float incomes like 19999.99.
    """

    @staticmethod
    def make_sample_data() -> pd.DataFrame:
        """
        8 accounts covering every boundary condition:
          id=1:  16000  → Low     (below 20k)
          id=2:  19999  → Low     (just below 20k boundary)
          id=3:  20000  → Average (exactly AT lower boundary)
          id=4:  35000  → Average (squarely in the middle)
          id=5:  50000  → Average (exactly AT upper boundary)
          id=6:  50001  → High    (just above 50k boundary)
          id=7:  80000  → High    (well above 50k)
          id=8: 120000  → High    (far above 50k)
        """
        return pd.DataFrame({
            "account_id": [1,      2,      3,      4,      5,      6,      7,      8     ],
            "income":     [16000,  19999,  20000,  35000,  50000,  50001,  80000,  120000],
        })

    @staticmethod
    def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:

        log.info("=== count_salary_categories ===")
        log.info("Input shape: %d rows x %d cols", *accounts.shape)
        log.debug("Input:\n%s\n", accounts.to_string(index=False))

        # ── Step 1: define boolean masks for each category ───────────
        # Each mask is a boolean Series — True for rows that belong to that bucket.
        low_mask     = accounts["income"] < 20000
        average_mask = (accounts["income"] >= 20000) & (accounts["income"] <= 50000)
        high_mask    = accounts["income"] > 50000

        log.debug("Mask counts — Low: %d  |  Average: %d  |  High: %d",
                  low_mask.sum(), average_mask.sum(), high_mask.sum())

        # Sanity check: every row should fall into exactly one category
        assert (low_mask | average_mask | high_mask).all(), \
            "Some rows were not assigned to any category"
        assert not (low_mask & average_mask).any(), \
            "Some rows matched both Low and Average"
        assert not (average_mask & high_mask).any(), \
            "Some rows matched both Average and High"
        log.debug("Sanity check passed: categories are exhaustive and mutually exclusive")

        # ── Step 2: build the result DataFrame directly ──────────────
        # Hardcoding all three rows guarantees zero-count categories are
        # included — groupby would silently drop empty buckets.
        # .sum() on a boolean Series counts the True values (= matching rows).
        result = pd.DataFrame({
            "category": ["Low Salary", "Average Salary", "High Salary"],
            "accounts_count": [
                low_mask.sum(),      # mask.sum() preferred over df[mask].shape[0]
                average_mask.sum(),
                high_mask.sum(),
            ],
        })

        log.info("Result:\n%s\n", result.to_string(index=False))
        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        assert list(result.columns) == ["category", "accounts_count"], \
            f"Wrong columns: {list(result.columns)}"
        assert len(result) == 3, \
            f"Expected exactly 3 rows, got {len(result)}"

        expected = pd.DataFrame({
            "category":      ["Low Salary", "Average Salary", "High Salary"],
            "accounts_count": [2,            3,                3            ],
        })

        # Sort by category so row order does not matter
        res = result.sort_values("category").reset_index(drop=True)
        exp = expected.sort_values("category").reset_index(drop=True)
        pd.testing.assert_frame_equal(res, exp, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = CountSalaryCategories()

    df = solver.make_sample_data()
    result = solver.count_salary_categories(df)
    solver.verify(result)

    # ── Show why & not 'and', and why parentheses matter ─────────────
    log.info("--- Operator precedence: why parentheses are required ---")
    s = pd.Series([15000, 30000, 60000])
    correct   = (s >= 20000) & (s <= 50000)
    log.info("Correct   (s >= 20000) & (s <= 50000) → %s", correct.tolist())
    # Uncomment the next line to see the ValueError 'and' raises on a Series:
    # wrong = s >= 20000 and s <= 50000

    # ── Show mask.sum() vs shape[0] are identical ─────────────────────
    log.info("--- mask.sum() vs df[mask].shape[0] vs len(df[mask]) ---")
    mask = df["income"] < 20000
    log.info("mask.sum()         = %d", mask.sum())
    log.info("df[mask].shape[0]  = %d", df[mask].shape[0])
    log.info("len(df[mask])      = %d", len(df[mask]))

    # ── Edge case: all accounts in one category ───────────────────────
    log.info("--- Edge case: all accounts are High Salary ---")
    all_high = pd.DataFrame({"account_id": [1, 2, 3], "income": [60000, 70000, 80000]})
    result2 = solver.count_salary_categories(all_high)
    log.info("Low and Average should be 0:\n%s\n", result2.to_string(index=False))

    # ── Edge case: empty accounts table ──────────────────────────────
    log.info("--- Edge case: empty accounts table ---")
    empty = pd.DataFrame(columns=["account_id", "income"])
    result3 = solver.count_salary_categories(empty)
    log.info("All categories should be 0:\n%s\n", result3.to_string(index=False))