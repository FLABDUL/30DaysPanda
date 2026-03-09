import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class ReplaceEmployeeId:
    """
    LeetCode 30 Days of Pandas – "Replace Employee ID With The Unique Identifier"
    ------------------------------------------------------------------------------
    PROBLEM:
        Two tables:
            employees    – columns: id, name
            employee_uni – columns: id, unique_id  (NOT every employee has a row here)

        Return a table with columns [unique_id, name] for ALL employees.
        If an employee has no unique_id, show NULL (NaN) in that column.
        Sort the result by name ascending.

    CORE CONCEPT: pd.merge with how='left'
        pd.merge(left, right, on='id', how='left')

        The `how` parameter controls which rows survive the join.
        Think of it as: "which table's rows are we committed to keeping?"

        ┌─────────────┬──────────────────────────────────────────────────────┐
        │ how=         │ behaviour                                            │
        ├─────────────┼──────────────────────────────────────────────────────┤
        │ 'inner'     │ ONLY rows whose key exists in BOTH tables (default)  │
        │             │ → employees with no unique_id would be DROPPED        │
        ├─────────────┼──────────────────────────────────────────────────────┤
        │ 'left'      │ ALL rows from the LEFT table are kept                │
        │             │ → every employee is kept; unmatched unique_id = NaN  │
        ├─────────────┼──────────────────────────────────────────────────────┤
        │ 'right'     │ ALL rows from the RIGHT table are kept               │
        │             │ → employees with no row in employee_uni are DROPPED  │
        ├─────────────┼──────────────────────────────────────────────────────┤
        │ 'outer'     │ ALL rows from BOTH tables; NaN wherever no match     │
        └─────────────┴──────────────────────────────────────────────────────┘

        This problem requires 'left' because the question says to return ALL
        employees — including those without a unique_id entry. An 'inner' join
        would silently drop them, producing a wrong answer.

    PANDAS GOTCHAS:
        • `on='id'` only works when both DataFrames have a column literally named 'id'.
          If the key columns had different names you would use:
          left_on='employee_id', right_on='emp_id'
        • After a left join, unmatched right-side columns (like unique_id) are NaN —
          not 0, not empty string — so downstream code must handle NaN explicitly
        • Column order after merge is: all left cols, then all right cols (minus the
          shared key). Always select explicitly rather than relying on position.
    """

    @staticmethod
    def make_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        employees has 5 rows.
        employee_uni only covers 3 of them — exercises all four join outcomes:
          - id=1: has a unique_id → matched row
          - id=2: has a unique_id → matched row
          - id=3: has a unique_id → matched row
          - id=4: no unique_id entry → left join keeps it with NaN
          - id=5: no unique_id entry → left join keeps it with NaN
        """
        employees = pd.DataFrame({
            "id":   [1, 2, 3, 4, 5],
            "name": ["Alice", "Charlie", "Bob", "Diana", "Eve"],
            # deliberately unsorted to confirm sort_values is doing its job
        })

        employee_uni = pd.DataFrame({
            "id":        [3, 1, 2],   # out of order to prove merge matches by value, not position
            "unique_id": [101, 102, 103],
        })

        log.debug("employees table:\n%s\n", employees.to_string(index=False))
        log.debug("employee_uni table:\n%s\n", employee_uni.to_string(index=False))
        return employees, employee_uni

    @staticmethod
    def replace_employee_id(
        employees: pd.DataFrame,
        employee_uni: pd.DataFrame,
    ) -> pd.DataFrame:

        log.info("=== replace_employee_id ===")
        log.info("employees shape:    %d rows x %d cols", *employees.shape)
        log.info("employee_uni shape: %d rows x %d cols", *employee_uni.shape)

        # ── Step 1: left join on 'id' ────────────────────────────────
        # Every row in `employees` (the LEFT table) is guaranteed to survive.
        # Rows in `employee_uni` (the RIGHT table) that have no matching employee
        # are discarded. Employees with no match in employee_uni get NaN for unique_id.
        merged = pd.merge(employees, employee_uni, on="id", how="left")

        log.debug("After left merge (all columns, before selection):\n%s\n",
                  merged.to_string(index=False))
        log.debug("Rows with NaN unique_id (no entry in employee_uni): %d",
                  merged["unique_id"].isna().sum())

        # ── Step 2: select and sort ──────────────────────────────────
        # Drop 'id' — the problem only wants unique_id and name.
        # sort_values is ascending by default; no need to pass ascending=True.
        result = merged[["unique_id", "name"]].sort_values(by=["name"]).reset_index(drop=True)

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))

        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        res = result.reset_index(drop=True)

        # Names sorted alphabetically: Alice, Bob, Charlie, Diana, Eve
        expected = pd.DataFrame({
            "unique_id": [102.0, 101.0, 103.0, None, None],  # float because NaN promotes int → float
            "name":      ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        })

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = ReplaceEmployeeId()

    employees, employee_uni = solver.make_sample_data()
    result = solver.replace_employee_id(employees, employee_uni)
    solver.verify(result)

    # ── Demonstrate why 'inner' would be wrong ───────────────────────
    log.info("--- Contrast: inner join drops unmatched employees ---")
    inner = pd.merge(employees, employee_uni, on="id", how="inner")
    inner_result = inner[["unique_id", "name"]].sort_values("name").reset_index(drop=True)
    log.info("inner join result (%d rows — Diana and Eve are GONE):\n%s\n",
             len(inner_result), inner_result.to_string(index=False))

    # ── Edge case: empty employee_uni ───────────────────────────────
    log.info("--- Edge case: no unique IDs exist at all ---")
    empty_uni = pd.DataFrame(columns=["id", "unique_id"])
    empty_result = solver.replace_employee_id(employees, empty_uni)
    log.info("All unique_id values are NaN: %s", empty_result["unique_id"].isna().all())

    # ── Edge case: empty employees ───────────────────────────────────
    log.info("--- Edge case: empty employees table ---")
    empty_emp = pd.DataFrame(columns=["id", "name"])
    empty_result2 = solver.replace_employee_id(empty_emp, employee_uni)
    log.info("Empty employees → %d rows (expected 0)", len(empty_result2))