import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class DepartmentHighestSalary:
    """
    LeetCode 30 Days of Pandas – "Department Highest Salary"
    ---------------------------------------------------------
    PROBLEM:
        Two tables:
            employee   – columns: id, name, salary, departmentId
            department – columns: id, name

        Return every employee whose salary is the highest in their department.
        Multiple employees can tie for the top salary in the same department.
        Output columns: Department, Employee, Salary.

    CORE CONCEPT: merge → groupby transform('max') → boolean mask filter

        STEP 1 — merge on mismatched key names (left_on / right_on)
            employee.merge(department, left_on='departmentId', right_on='id')

            Both tables have a column called 'id' but they mean different things:
              employee.id    → the employee's own ID
              department.id  → the department's ID (= employee.departmentId)

            Because the join key has DIFFERENT names in each table we cannot
            use on='id' (that would match employee.id to department.id — wrong).
            Instead:
              left_on='departmentId'  – the key column in the LEFT table
              right_on='id'           – the matching key in the RIGHT table

            After the merge, pandas auto-suffixes ALL clashing column names:
              id_x   → employee.id
              id_y   → department.id
              name_x → employee.name
              name_y → department.name

        STEP 2 — groupby + transform('max')
            df.groupby('departmentId')['salary'].transform('max')

            This is the most important concept in this problem.

            transform vs agg:
            ┌──────────────┬──────────────────────────────────────────────────┐
            │ .agg('max')  │ COLLAPSES the group → one row per department     │
            │              │ result has fewer rows than the input             │
            ├──────────────┼──────────────────────────────────────────────────┤
            │ .transform() │ BROADCASTS the result back → same length as input│
            │              │ every employee row gets the max salary for THEIR  │
            │              │ department injected alongside their own salary    │
            └──────────────┴──────────────────────────────────────────────────┘

            After transform, each row has TWO salary values side by side:
              df['salary']                              → the employee's own salary
              df.groupby(...)['salary'].transform('max') → their department's max

            This makes the filter a simple row-wise equality comparison.

        STEP 3 — boolean mask: keep rows where salary == department max
            df[df['salary'] == df.groupby(...)['salary'].transform('max')]

            Ties are preserved naturally — if three people share the max
            salary all three rows survive because == matches all of them.

        STEP 4 — column selection + rename
            df[['name_y','name_x','salary']].rename(...)

            Picks the three required columns (note _x/_y suffixes) and renames
            them to the expected output names.

    PANDAS GOTCHAS:
        • left_on/right_on keep BOTH key columns in the output (departmentId
          AND department's id). on= keeps only one shared column. Fine here
          because we select only the columns we want at the end.
        • Auto-suffixing (_x, _y) applies to ALL clashing column names, not
          just the join key. Always inspect the merged DataFrame before
          selecting columns — it is easy to grab the wrong 'name'.
        • transform returns a Series aligned to the original DataFrame index,
          so it can be used directly in a boolean comparison — no reset_index.
        • If all employees in a department earn the same salary, all of them
          are returned. The == comparison handles this correctly.
    """

    @staticmethod
    def make_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        3 departments, 5 employees, designed to test:
          - IT   (dept 1): Alice=90k, Bob=70k        → Alice wins alone
          - Sales (dept 2): Charlie=80k, Diana=80k   → TIE: both returned
          - HR   (dept 3): Eve=60k                   → only employee, always wins
        """
        employee = pd.DataFrame({
            "id":           [1,       2,     3,         4,       5     ],
            "name":         ["Alice", "Bob", "Charlie", "Diana", "Eve" ],
            "salary":       [90000,   70000, 80000,     80000,   60000 ],
            "departmentId": [1,       1,     2,         2,       3     ],
        })

        department = pd.DataFrame({
            "id":   [1,    2,       3   ],
            "name": ["IT", "Sales", "HR"],
        })

        log.debug("employee:\n%s\n",   employee.to_string(index=False))
        log.debug("department:\n%s\n", department.to_string(index=False))
        return employee, department

    @staticmethod
    def department_highest_salary(
        employee: pd.DataFrame,
        department: pd.DataFrame,
    ) -> pd.DataFrame:

        log.info("=== department_highest_salary ===")
        log.info("employee: %d rows  |  department: %d rows",
                 len(employee), len(department))

        # ── Step 1: merge on mismatched key names ────────────────────
        # left_on/right_on needed because the join key is named differently
        # in each table. Clashing column names get _x (left) / _y (right).
        df = employee.merge(department, left_on="departmentId", right_on="id")

        log.debug("After merge (note _x/_y suffixes on clashing columns):\n%s\n",
                  df.to_string(index=False))

        # ── Step 2: broadcast department max salary to every row ─────
        # transform('max') keeps the same number of rows as df — every
        # employee row now has its department's max salary alongside it.
        dept_max = df.groupby("departmentId")["salary"].transform("max")

        log.debug("Per-employee department-max salary (via transform):\n%s\n",
                  pd.concat(
                      [df[["name_x", "departmentId", "salary"]],
                       dept_max.rename("dept_max")],
                      axis=1
                  ).to_string(index=False))

        # ── Step 3: filter — keep rows at the department maximum ─────
        mask = df["salary"] == dept_max
        log.debug("Employees at their department's maximum: %d / %d",
                  mask.sum(), len(df))

        # ── Step 4: select and rename ────────────────────────────────
        # name_y = department name (right table)
        # name_x = employee name  (left table)
        result = (
            df[mask][["name_y", "name_x", "salary"]]
            .rename(columns={"name_y": "Department",
                             "name_x": "Employee",
                             "salary": "Salary"})
            .reset_index(drop=True)
        )

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))
        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        res = result.sort_values(["Department", "Employee"]).reset_index(drop=True)

        expected = pd.DataFrame({
            "Department": ["HR",  "IT",    "Sales",   "Sales" ],
            "Employee":   ["Eve", "Alice", "Charlie", "Diana" ],
            "Salary":     [60000, 90000,   80000,     80000   ],
        })

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = DepartmentHighestSalary()

    employee, department = solver.make_sample_data()
    result = solver.department_highest_salary(employee, department)
    solver.verify(result)

    # ── Demonstrate transform vs agg ─────────────────────────────────
    log.info("--- transform('max') vs agg('max') — the key difference ---")
    df_merged = employee.merge(department, left_on="departmentId", right_on="id")

    agg_result = df_merged.groupby("departmentId")["salary"].agg("max")
    log.info("agg('max') COLLAPSES → %d rows (one per dept):\n%s\n",
             len(agg_result), agg_result.to_string())

    transform_result = df_merged.groupby("departmentId")["salary"].transform("max")
    log.info("transform('max') BROADCASTS → %d rows (one per employee):\n%s\n",
             len(transform_result), transform_result.to_string())

    # ── Edge case: entire department tied at max salary ───────────────
    log.info("--- Edge case: all employees in one dept on same salary ---")
    one_dept_emp = pd.DataFrame({
        "id":           [1,     2,     3    ],
        "name":         ["X",   "Y",   "Z"  ],
        "salary":       [50000, 90000, 90000],  # Y and Z tie for max
        "departmentId": [1,     1,     1    ],
    })
    result2 = solver.department_highest_salary(one_dept_emp, department)
    log.info("Two-way tie — both returned:\n%s\n", result2.to_string(index=False))

    # ── Edge case: single employee per department ─────────────────────
    log.info("--- Edge case: one employee per department ---")
    one_each = pd.DataFrame({
        "id":           [1,     2,     3    ],
        "name":         ["A",   "B",   "C"  ],
        "salary":       [10000, 20000, 30000],
        "departmentId": [1,     2,     3    ],
    })
    result3 = solver.department_highest_salary(one_each, department)
    log.info("Every employee is their dept's max:\n%s\n", result3.to_string(index=False))