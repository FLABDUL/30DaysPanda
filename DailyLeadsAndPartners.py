import pandas as pd
import logging

# Configure logging so we can see step-by-step what's happening at runtime
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


class DailyLeadsAndPartners:
    """
    LeetCode 30 Days of Pandas – "Daily Leads and Partners"
    --------------------------------------------------------
    PROBLEM:
        Given a table `daily_sales` with columns:
            date_id    – the sale date
            make_name  – car brand
            lead_id    – ID of a sales lead  (may repeat across rows)
            partner_id – ID of a partner     (may repeat across rows)

        Return a DataFrame with one row per (date_id, make_name) pair showing:
            unique_leads    – number of DISTINCT lead_id values
            unique_partners – number of DISTINCT partner_id values

    CORE CONCEPT: groupby + agg with 'nunique'
        • groupby(['date_id','make_name'])
              → splits the table into one bucket per (date, brand) combination
        • .agg({'lead_id':'nunique', 'partner_id':'nunique'})
              → for each bucket, counts unique values in each column
              → 'nunique' is shorthand for "number of unique" (ignores NaN)
        • .reset_index()
              → turns the group keys back into regular columns
    """

    # ------------------------------------------------------------------
    # 1. Build realistic sample data
    # ------------------------------------------------------------------
    @staticmethod
    def make_sample_data() -> pd.DataFrame:
        """
        Construct a small daily_sales DataFrame that exercises all edge cases:
          - duplicate lead_id on the same (date, make)  → should still count as 1
          - duplicate partner_id on the same (date, make) → same
          - multiple dates and multiple makes
        """
        data = {
            "date_id": [
                "2020-12-8", "2020-12-8", "2020-12-8",   # three rows, same date+make
                "2020-12-8", "2020-12-8",                 # same date, different make
                "2020-12-7", "2020-12-7",                 # different date
            ],
            "make_name": [
                "toyota", "toyota", "toyota",
                "honda",  "honda",
                "toyota", "toyota",
            ],
            "lead_id": [
                0, 1, 1,   # lead 1 appears twice → unique count = 2
                0, 2,
                0, 0,      # lead 0 appears twice → unique count = 1
            ],
            "partner_id": [
                1, 0, 0,   # partner 0 appears twice → unique count = 2
                2, 2,      # partner 2 appears twice → unique count = 1
                0, 1,
            ],
        }
        df = pd.DataFrame(data)
        log.debug("Sample data created:\n%s\n", df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # 2. The solution (with step-by-step logging)
    # ------------------------------------------------------------------
    @staticmethod
    def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:

        log.info("=== daily_leads_and_partners ===")
        log.info("Input shape: %s rows × %s cols", *daily_sales.shape)

        # ── Step 1: groupby ──────────────────────────────────────────
        # Create a GroupBy object (no computation yet – lazy).
        # Each group = all rows that share the same (date_id, make_name).
        grouped_obj = daily_sales.groupby(["date_id", "make_name"])

        log.debug("Number of groups (unique date+make pairs): %d",
                  grouped_obj.ngroups)

        # ── Step 2: agg with nunique ─────────────────────────────────
        # .agg() applies one or more aggregation functions per column.
        # 'nunique' counts distinct non-null values.
        # Equivalent long form:
        #   .agg(unique_leads=('lead_id','nunique'),
        #        unique_partners=('partner_id','nunique'))
        grouped = grouped_obj.agg(
            {
                "lead_id":    "nunique",   # → column still named 'lead_id'
                "partner_id": "nunique",   # → column still named 'partner_id'
            }
        ).reset_index()  # date_id & make_name become normal columns again

        log.debug("After groupby+agg (before rename):\n%s\n",
                  grouped.to_string(index=False))

        # ── Step 3: rename columns ───────────────────────────────────
        # The aggregated columns kept the original names; rename for clarity.
        grouped.columns = ["date_id", "make_name", "unique_leads", "unique_partners"]

        log.info("Output shape: %s rows × %s cols", *grouped.shape)
        log.info("Result:\n%s\n", grouped.to_string(index=False))

        return grouped

    # ------------------------------------------------------------------
    # 3. Verify output is correct
    # ------------------------------------------------------------------
    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        """
        Hard-coded expected values for the sample data.
        Raises AssertionError if anything is wrong.
        """
        # Sort both sides the same way so row order doesn't matter
        res = result.sort_values(["date_id", "make_name"]).reset_index(drop=True)

        expected = pd.DataFrame({
            "date_id":         ["2020-12-7", "2020-12-8", "2020-12-8"],
            "make_name":       ["toyota",    "honda",     "toyota"],
            "unique_leads":    [1,           2,           2],
            "unique_partners": [2,           1,           2],
        })

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    solver = DailyLeadsAndPartners()

    # Build sample data
    daily_sales_df = solver.make_sample_data()

    # Run solution
    result_df = solver.daily_leads_and_partners(daily_sales_df)

    # Confirm correctness
    solver.verify(result_df)

    # ── Bonus: try it on an empty DataFrame ─────────────────────────
    log.info("--- Edge case: empty input ---")
    empty_input = pd.DataFrame(
        columns=["date_id", "make_name", "lead_id", "partner_id"]
    )
    empty_result = solver.daily_leads_and_partners(empty_input)
    log.info("Empty input produces %d rows (expected 0)", len(empty_result))