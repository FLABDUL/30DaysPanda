# Copilot Instructions – LeetCode 30 Days of Pandas

When I paste a LeetCode pandas solution, wrap it in a **runnable, self-contained Python class** that follows the exact pattern below.

---

## Class Structure (always follow this order)

```python
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class <ProblemNameInPascalCase>:
    """
    LeetCode 30 Days of Pandas – "<Exact Problem Title>"
    -----------------------------------------------------
    PROBLEM:
        <copy the problem statement here – table schema + what to return>

    CORE CONCEPT: <the main pandas technique being used>
        • <bullet explaining step 1 of the solution>
        • <bullet explaining step 2>
        • <and so on – explain WHY each step is needed, not just what it does>

    PANDAS GOTCHAS:
        • <any non-obvious behaviour to watch out for>
    """

    @staticmethod
    def make_sample_data() -> pd.DataFrame:
        """
        Build a small DataFrame that covers:
          - the happy path (normal rows)
          - at least one duplicate / edge case that the solution must handle correctly
          - multiple groups if groupby is involved
        Add a comment on each tricky row explaining what it tests.
        """
        ...

    @staticmethod
    def solution(<input>: pd.DataFrame) -> pd.DataFrame:
        """
        Log every meaningful step with log.debug() or log.info() so the
        learner can see the transformation at runtime.

        Step pattern:
          1. log.info("=== <method name> ===")
          2. log.info("Input shape: %s rows x %s cols", *df.shape)
          3. For each transformation: log.debug("<what just happened>:\n%s\n", result)
          4. log.info("Output shape / result summary")
        """
        ...

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        """
        Hard-code the expected output for make_sample_data().
        Use pd.testing.assert_frame_equal(..., check_dtype=False).
        Log "✅  All assertions passed" on success.
        """
        ...


if __name__ == "__main__":
    solver = <ProblemNameInPascalCase>()
    df = solver.make_sample_data()
    result = solver.solution(df)
    solver.verify(result)

    # Always add at least one edge-case run (empty DataFrame, single row, etc.)
    log.info("--- Edge case: empty input ---")
    ...
```

---

## Rules

### Naming
- Class name → `PascalCase` of the problem title (e.g. `FindUsersWithValidEmails`)
- Method that wraps the solution → use the original LeetCode function name exactly
- Input parameter names → match the LeetCode schema exactly

### `make_sample_data`
- Cover the **minimum set of rows** that proves the solution works
- Include rows that test duplicate handling, NaN handling, boundary conditions — whichever are relevant to *this* problem
- Comment every non-obvious row: `# tests: <what this row exercises>`

### Logging inside `solution`
- `log.info` for high-level milestones (start, end, shape changes)
- `log.debug` for intermediate DataFrames — print with `.to_string(index=False)`
- Never log inside a tight loop; log the result *after* the operation

### `verify`
- Sort both `result` and `expected` the same way before comparing (so row order never causes false failures)
- Use `pd.testing.assert_frame_equal(..., check_dtype=False)` unless dtype is part of the problem

### Docstring – CORE CONCEPT section
Explain **why** the pandas API works the way it does, not just what it is called.
Examples of good explanations:
- "`groupby` is lazy — it creates buckets but does no computation until `.agg()` is called"
- "`nunique` ignores NaN; `count` counts non-null rows — they are NOT the same"
- "`merge` defaults to inner join; use `how='left'` to keep rows with no match"

### What NOT to do
- Do not split the class across multiple files
- Do not add `__init__` unless state is genuinely needed
- Do not use `print()` — always use `log`
- Do not explain the solution in prose outside the docstring; let the code and logs speak

---

## Example trigger

When I say something like:

> "write a class for this leet problem: `def top_travellers(...)`"

…produce the full class immediately, following every rule above, with no extra explanation outside the class itself.