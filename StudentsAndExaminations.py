import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class StudentsAndExaminations:
    """
    LeetCode 30 Days of Pandas – "Students and Examinations"
    ---------------------------------------------------------
    PROBLEM:
        Three tables:
            students     – columns: student_id, student_name
            subjects     – columns: subject_name
            examinations – columns: student_id, subject_name  (one row per exam sat)

        Return ONE row per (student, subject) combination — even if the student
        never sat that exam — showing how many times they attended.
        Sort by student_id ASC, then subject_name ASC.

    CORE CONCEPT: cross join → groupby count → left join → fillna
        This solution is a three-step pipeline, each step building on the last.

        STEP 1 — cross join (how='cross')
            students.merge(subjects, how='cross')
            A cross join produces the CARTESIAN PRODUCT: every student paired
            with every subject. If there are 3 students and 4 subjects, you get
            3 × 4 = 12 rows. This guarantees a row exists for every combination
            even before we know whether any exams were sat.

            ┌─────────────┬─────────────────────────────────────────────────┐
            │ how=        │ behaviour                                        │
            ├─────────────┼─────────────────────────────────────────────────┤
            │ 'inner'     │ match rows by key (requires on= or left/right_on)│
            │ 'left'      │ keep all left rows; NaN for unmatched right      │
            │ 'right'     │ keep all right rows; NaN for unmatched left      │
            │ 'outer'     │ keep all rows from both; NaN for any unmatched   │
            │ 'cross'     │ NO key needed — pairs every left row with every  │
            │             │ right row; result has len(left) × len(right) rows│
            └─────────────┴─────────────────────────────────────────────────┘
            'cross' is the only how value that does not use a join key at all.

        STEP 2 — groupby + named agg (count attended exams)
            examinations.groupby(['student_id','subject_name'])
                         .agg(attended_exams=('subject_name','count'))
            Each row in examinations is one exam sitting. Counting rows per
            (student, subject) group gives the number of times they attended.
            This produces a SPARSE table — only combinations that appear at
            least once are present.

        STEP 3 — left join the count onto the full combination grid
            two_df.merge(exam_count, on=['student_id','subject_name'], how='left')
            The cross-joined grid is the LEFT table (all combinations guaranteed).
            exam_count is the RIGHT table (sparse — only attended ones).
            Combinations with no exam sittings get NaN in attended_exams.

        STEP 4 — fillna(0)
            NaN means "no row existed in examinations" which means 0 attendances.
            Always fill AFTER the join — never before — or you'd lose the signal
            that a count is truly missing vs genuinely zero.

    PANDAS GOTCHAS:
        • 'cross' was added in pandas 1.2.0 — it will raise on older versions
        • After a left join, integer columns become float if NaN is introduced
          (same promotion as the previous problem). fillna(0) leaves them as
          float (0.0) unless you chain .astype(int) — acceptable for LeetCode
        • groupby silently drops rows where the key columns contain NaN.
          Here that is fine because student_id and subject_name are never null.
        • sort_values default is ascending=True for all keys; no need to state it
    """

    @staticmethod
    def make_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        3 students × 2 subjects = 6 expected output rows.

        Exam attendances are deliberately sparse to exercise every case:
          - student 1 / Math:    attended twice  → count = 2
          - student 1 / Physics: never attended  → count = 0  (the critical NaN case)
          - student 2 / Math:    attended once   → count = 1
          - student 2 / Physics: attended once   → count = 1
          - student 3 / Math:    never attended  → count = 0
          - student 3 / Physics: never attended  → count = 0
        """
        students = pd.DataFrame({
            "student_id":   [1, 2, 3],
            "student_name": ["Alice", "Bob", "Charlie"],
        })

        subjects = pd.DataFrame({
            "subject_name": ["Math", "Physics"],
        })

        examinations = pd.DataFrame({
            "student_id":   [1, 1, 2, 2],       # students 3 has no rows at all
            "subject_name": ["Math", "Math", "Math", "Physics"],
            # student 1 sat Math twice; student 1 never sat Physics
        })

        log.debug("students:\n%s\n",     students.to_string(index=False))
        log.debug("subjects:\n%s\n",     subjects.to_string(index=False))
        log.debug("examinations:\n%s\n", examinations.to_string(index=False))
        return students, subjects, examinations

    @staticmethod
    def students_and_examinations(
        students: pd.DataFrame,
        subjects: pd.DataFrame,
        examinations: pd.DataFrame,
    ) -> pd.DataFrame:

        log.info("=== students_and_examinations ===")
        log.info("students: %d rows  |  subjects: %d rows  |  examinations: %d rows",
                 len(students), len(subjects), len(examinations))

        # ── Step 1: cross join ───────────────────────────────────────
        # Produces every (student, subject) pair — guaranteed complete grid.
        # No on= needed; 'cross' ignores keys entirely.
        two_df = students.merge(subjects, how="cross")

        log.debug("After cross join (%d students × %d subjects = %d rows):\n%s\n",
                  len(students), len(subjects), len(two_df),
                  two_df.to_string(index=False))

        # ── Step 2: count exam sittings per (student, subject) ───────
        # This is SPARSE — only pairs that appear in examinations are present.
        exam_count = (
            examinations
            .groupby(["student_id", "subject_name"])
            .agg(attended_exams=("subject_name", "count"))
            .reset_index()
        )

        log.debug("Exam counts (sparse — only attended combinations):\n%s\n",
                  exam_count.to_string(index=False))
        log.debug("Combinations present in exam_count: %d  "
                  "(missing %d pairs that were never attended)",
                  len(exam_count),
                  len(two_df) - len(exam_count))

        # ── Step 3: left join — attach counts to the full grid ───────
        # two_df is LEFT (complete), exam_count is RIGHT (sparse).
        # Unmatched rows in two_df get NaN for attended_exams.
        all_df = two_df.merge(exam_count, on=["student_id", "subject_name"], how="left")

        log.debug("After left join (NaN where student never attended):\n%s\n",
                  all_df.to_string(index=False))
        log.debug("Rows still showing NaN (never attended): %d",
                  all_df["attended_exams"].isna().sum())

        # ── Step 4: fill NaN → 0, select columns, sort ───────────────
        # NaN means the pair was absent from examinations → 0 sittings.
        all_df["attended_exams"] = all_df["attended_exams"].fillna(0).astype(int)

        result = (
            all_df[["student_id", "student_name", "subject_name", "attended_exams"]]
            .sort_values(by=["student_id", "subject_name"])
            .reset_index(drop=True)
        )

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))
        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        res = result.reset_index(drop=True)

        expected = pd.DataFrame({
            "student_id":    [1, 1, 2, 2, 3, 3],
            "student_name":  ["Alice", "Alice", "Bob", "Bob", "Charlie", "Charlie"],
            "subject_name":  ["Math", "Physics", "Math", "Physics", "Math", "Physics"],
            "attended_exams":[2, 0, 1, 1, 0, 0],
        })

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = StudentsAndExaminations()

    students, subjects, examinations = solver.make_sample_data()
    result = solver.students_and_examinations(students, subjects, examinations)
    solver.verify(result)

    # ── Demonstrate why skipping the cross join breaks things ────────
    log.info("--- Contrast: skipping cross join (direct left join) ---")
    # If we skipped the cross join and just left-joined students onto exam_count
    # directly, students with ZERO exams across ALL subjects disappear entirely.
    exam_count_only = (
        examinations
        .groupby(["student_id", "subject_name"])
        .agg(attended_exams=("subject_name", "count"))
        .reset_index()
    )
    naive = students.merge(exam_count_only, on="student_id", how="left")
    log.info("Naive approach gives %d rows — Charlie (0 exams) is misrepresented:\n%s\n",
             len(naive), naive.to_string(index=False))

    # ── Edge case: empty examinations ────────────────────────────────
    log.info("--- Edge case: no exams sat by anyone ---")
    empty_exams = pd.DataFrame(columns=["student_id", "subject_name"])
    result2 = solver.students_and_examinations(students, subjects, empty_exams)
    log.info("All attended_exams are 0: %s", (result2["attended_exams"] == 0).all())

    # ── Edge case: single student, single subject ─────────────────────
    log.info("--- Edge case: 1 student, 1 subject, 1 exam ---")
    r3 = solver.students_and_examinations(
        pd.DataFrame({"student_id": [99], "student_name": ["Zara"]}),
        pd.DataFrame({"subject_name": ["Biology"]}),
        pd.DataFrame({"student_id": [99], "subject_name": ["Biology"]}),
    )
    log.info("Result:\n%s\n", r3.to_string(index=False))