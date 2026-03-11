import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class FindManagers:
    """
    LeetCode 30 Days of Pandas – "Managers with at Least 5 Direct Reports"
    -----------------------------------------------------------------------
    PROBLEM:
        Given a table `employee` with columns:
            id         – employee's unique ID
            name       – employee's name
            department – department they belong to
            managerId  – the id of their manager (NULL if they have no manager)

        Return the name of every manager who has AT LEAST 5 direct reports.

    CORE CONCEPT: groupby + count → boolean mask → isin filter

        STEP 1 — groupby('managerId')['id'].count()
            Groups every row by the managerId column, then counts the 'id'
            values in each group. Each row in the original table represents one
            employee, so the count per group = number of direct reports.
            Returns a Series indexed by managerId, values are the report counts.

            groupby vs value_counts for this task:
            ┌──────────────────────┬─────────────────────────────────────────┐
            │ groupby + count      │ explicit; scales to multi-column agg;   │
            │                      │ NaN keys dropped by default (good here) │
            ├──────────────────────┼─────────────────────────────────────────┤
            │ value_counts()       │ shorter one-liner for single-col counts; │
            │                      │ also drops NaN by default               │
            └──────────────────────┴─────────────────────────────────────────┘
            groupby makes the intent explicit: "group by manager, count reports."

        STEP 2 — filter the counts Series with a boolean mask
            counts[counts >= 5]
            Boolean mask on a Series works identically to a DataFrame:
            keep only entries where the count is 5 or more.
            .index pulls out the managerId values that qualify.

        STEP 3 — isin() to filter the employee table
            employee['id'].isin(qualifying_manager_ids)
            isin() returns a boolean Series: True wherever the employee's id
            is in our set of qualifying manager ids.
            This is the lookup step that converts manager ids → manager names.

    PANDAS GOTCHAS:
        • value_counts() silently drops NaN by default (dropna=True).
          That is exactly what we want — NULL managerId means "no manager",
          so it should never be counted as a manager with reports.
        • counts[counts >= 5].index gives an Index object, not a list.
          isin() accepts either, so no conversion is needed.
        • The result is a DataFrame with only the 'name' column — wrap in [[]]
          to return a DataFrame rather than a Series.
        • An employee can be both a manager and have their own manager.
          The solution handles this correctly because it checks employee['id']
          (who they ARE) against managerId counts (who manages others).
    """

    @staticmethod
    def make_sample_data() -> pd.DataFrame:
        """
        7 employees structured to cover every important case:
          - id=1 (Alice):   managerId=None  → no manager; has 5 direct reports → IN result
          - id=2 (Bob):     reports to 1;   has 2 direct reports (6,7)         → NOT in result
          - id=3 (Charlie): reports to 1;   has 0 direct reports               → NOT in result
          - id=4 (Diana):   reports to 1;   has 0 direct reports               → NOT in result
          - id=5 (Eve):     reports to 1;   has 0 direct reports               → NOT in result
          - id=6 (Frank):   reports to 2;   has 0 direct reports               → NOT in result
          - id=7 (Grace):   reports to 2;   has 0 direct reports               → NOT in result
          - id=8 (Hank):    reports to 1 → Alice reaches exactly 5 direct reports
        """
        return pd.DataFrame({
            "id":         [1,    2,    3,    4,    5,    6,    7,    8   ],
            "name":       ["Alice","Bob","Charlie","Diana","Eve","Frank","Grace","Hank"],
            "department": ["A",  "A",  "B",  "B",  "C",  "A",  "B",  "C"  ],
            "managerId":  [None,  1,    1,    1,    1,    2,    2,    1   ],
            # Alice has 5 direct reports: Bob, Charlie, Diana, Eve, Hank
            # Bob has 2 direct reports: Frank, Grace
        })

    @staticmethod
    def find_managers(employee: pd.DataFrame) -> pd.DataFrame:

        log.info("=== find_managers ===")
        log.info("Input shape: %d rows x %d cols", *employee.shape)

        # ── Step 1: count how many employees report to each manager ──
        # groupby('managerId') buckets every row by who they report to.
        # ['id'].count() counts non-null id values in each bucket = direct reports.
        # NaN managerId rows (employees with no manager) form no bucket — dropped.
        counts = employee.groupby("managerId")["id"].count()

        log.debug("Direct report counts per manager id:\n%s\n", counts.to_string())

        # ── Step 2: find manager ids with >= 5 direct reports ────────
        # Boolean mask on the Series, then pull the qualifying index values.
        qualifying_ids = counts[counts >= 5].index

        log.debug("Manager ids with >= 5 direct reports: %s", qualifying_ids.tolist())

        # ── Step 3: look up those ids in the employee table ──────────
        # isin() checks each employee's id against the qualifying set.
        # [['name']] returns a DataFrame (one column) instead of a Series.
        result = employee[employee["id"].isin(qualifying_ids)][["name"]]

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))

        return result.reset_index(drop=True)

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        res = result.sort_values("name").reset_index(drop=True)

        expected = pd.DataFrame({"name": ["Alice"]})

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = FindManagers()

    df = solver.make_sample_data()
    result = solver.find_managers(df)
    solver.verify(result)

    # ── Demonstrate groupby vs value_counts — same result ────────────
    log.info("--- groupby + count vs value_counts — identical output ---")
    counts_gb = df.groupby("managerId")["id"].count().sort_values(ascending=False)
    counts_vc = df["managerId"].value_counts()
    log.info("groupby count:\n%s\n",   counts_gb.to_string())
    log.info("value_counts():\n%s\n",  counts_vc.to_string())

    # ── Edge case: nobody has 5+ reports ─────────────────────────────
    log.info("--- Edge case: no manager meets the threshold ---")
    small = pd.DataFrame({
        "id":        [1, 2, 3],
        "name":      ["X", "Y", "Z"],
        "department":["A", "A", "A"],
        "managerId": [None, 1, 1],   # manager 1 has only 2 reports
    })
    small_result = solver.find_managers(small)
    log.info("Result rows (expected 0): %d", len(small_result))

    # ── Edge case: multiple managers qualify ──────────────────────────
    log.info("--- Edge case: two managers both qualify ---")
    multi = pd.DataFrame({
        "id":        range(1, 13),
        "name":      [f"Emp{i}" for i in range(1, 13)],
        "department":["A"] * 12,
        "managerId": [None, None,        # managers 1 and 2 have no manager
                      1, 1, 1, 1, 1,    # 5 reports to manager 1
                      2, 2, 2, 2, 2],   # 5 reports to manager 2
    })
    multi_result = solver.find_managers(multi)
    log.info("Both managers found:\n%s\n", multi_result.to_string(index=False))