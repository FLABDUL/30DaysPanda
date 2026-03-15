import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class SalesPerson:
    """
    LeetCode 30 Days of Pandas – "Sales Person"
    --------------------------------------------
    PROBLEM:
        Three tables:
            sales_person – columns: sales_id, name, salary, commission_rate, hire_date
            company      – columns: com_id, name, city
            orders       – columns: order_id, order_date, com_id, sales_id, amount

        Return the name of every sales person who has NO order with the company
        named 'RED'.

    CORE CONCEPT: inner join to find the guilty → ~ isin() to exclude them

        The solution works by EXCLUSION — find everyone who DID deal with RED,
        then return everyone NOT in that set. This is a classic anti-pattern:

        STEP 1 — isolate RED's com_id
            company[company['name'] == 'RED']
            Boolean mask on the company table. Returns only the row(s) for RED.
            This is passed as the RIGHT table into the merge so the join
            automatically restricts to RED's orders only.

        STEP 2 — inner join orders with RED
            pd.merge(left=orders, right=company[company['name']=='RED'],
                     on='com_id', how='inner')
            Only orders whose com_id matches RED survive an inner join.
            The result is a table of every order ever placed with RED.
            Pulling ['sales_id'].unique() from this gives the IDs of every
            sales person who has dealt with RED at least once.

            WHY inner join here (not left)?
            ┌────────────┬──────────────────────────────────────────────────┐
            │ how=       │ effect on this merge                             │
            ├────────────┼──────────────────────────────────────────────────┤
            │ 'inner'    │ only orders matching RED's com_id survive        │
            │            │ → gives us exactly the orders placed with RED    │
            ├────────────┼──────────────────────────────────────────────────┤
            │ 'left'     │ ALL orders survive; non-RED rows get NaN com_id  │
            │            │ → we'd need an extra dropna step; inner is cleaner│
            └────────────┴──────────────────────────────────────────────────┘

        STEP 3 — ~ isin() to invert the match
            ~sales_person['sales_id'].isin(red_sales_ids)
            isin() returns True wherever a sales_id IS in the guilty set.
            The ~ (tilde) is the bitwise NOT operator for boolean Series —
            it flips every True → False and False → True.
            Result: True only for sales people who NEVER dealt with RED.

            ~ is NOT the same as != here:
            • != 'value'  compares element-by-element against one value
            • ~isin([...]) inverts a membership test against a whole set
            Always use ~isin() when excluding a dynamic set of values.

    PANDAS GOTCHAS:
        • .unique() returns a numpy array, not a list — isin() accepts both
        • If 'RED' doesn't exist in company, the inner join returns an empty
          DataFrame, red_sales_ids is empty, and ALL sales people are returned
          (correct behaviour — nobody dealt with a non-existent company)
        • The solution uses the column 'name' from sales_person; after the
          inner join the merged table also has a 'name' column from company.
          Column selection [['name']] at the end is applied to sales_person
          directly, so there is no ambiguity — but be careful if you ever
          restructure the chain.
    """

    @staticmethod
    def make_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        4 sales people, 3 companies (RED, BLUE, YELLOW), 5 orders.

          - Alice  (id=1): has order with RED (order 1) and BLUE (order 2) → EXCLUDED
          - Bob    (id=2): has order with RED only (order 3)               → EXCLUDED
          - Charlie(id=3): has order with BLUE only (order 4)              → INCLUDED
          - Diana  (id=4): has order with YELLOW only (order 5)            → INCLUDED
          - Eve    (id=5): has NO orders at all                            → INCLUDED
        """
        sales_person = pd.DataFrame({
            "sales_id":         [1,       2,     3,         4,        5    ],
            "name":             ["Alice", "Bob", "Charlie", "Diana",  "Eve"],
            "salary":           [60000,   50000, 45000,     55000,    40000],
            "commission_rate":  [0.1,     0.08,  0.09,      0.07,     0.06 ],
            "hire_date":        ["2018-01-01"] * 5,
        })

        company = pd.DataFrame({
            "com_id": [1,     2,      3       ],
            "name":   ["RED", "BLUE", "YELLOW"],
            "city":   ["NYC", "LA",   "Chicago"],
        })

        orders = pd.DataFrame({
            "order_id":   [1,  2,  3,  4,  5 ],
            "order_date": ["2023-01-01"] * 5,
            "com_id":     [1,  2,  1,  2,  3 ],   # com_id 1 = RED
            "sales_id":   [1,  1,  2,  3,  4 ],   # Alice→RED, Bob→RED, Charlie→BLUE, Diana→YELLOW
            "amount":     [100, 200, 150, 300, 250],
        })

        log.debug("sales_person:\n%s\n", sales_person.to_string(index=False))
        log.debug("company:\n%s\n",      company.to_string(index=False))
        log.debug("orders:\n%s\n",       orders.to_string(index=False))
        return sales_person, company, orders

    @staticmethod
    def sales_person(
        sales_person: pd.DataFrame,
        company: pd.DataFrame,
        orders: pd.DataFrame,
    ) -> pd.DataFrame:

        log.info("=== sales_person ===")
        log.info("sales_person: %d rows  |  company: %d rows  |  orders: %d rows",
                 len(sales_person), len(company), len(orders))

        # ── Step 1: isolate RED ──────────────────────────────────────
        # Boolean mask produces only the RED row(s).
        # Using this as the right table in the join restricts matches to RED.
        red_company = company[company["name"] == "RED"]

        log.debug("RED company row(s):\n%s\n", red_company.to_string(index=False))

        # ── Step 2: inner join orders with RED ───────────────────────
        # Only orders whose com_id matches RED's com_id survive.
        # Result = every order ever placed with RED, including which sales_id made it.
        red_orders = pd.merge(
            left=orders,
            right=red_company,
            on="com_id",
            how="inner",
        )

        log.debug("Orders placed with RED (after inner join):\n%s\n",
                  red_orders.to_string(index=False))

        # ── Step 3: extract the guilty sales_ids ────────────────────
        # unique() removes duplicates — a sales person who placed 3 orders
        # with RED should still appear only once in the exclusion set.
        red_sales_ids = red_orders["sales_id"].unique()

        log.debug("Sales IDs who dealt with RED: %s", red_sales_ids.tolist())

        # ── Step 4: ~ isin() — exclude the guilty ───────────────────
        # isin() → True where sales_id IS in the guilty set
        # ~       → flip: True where sales_id is NOT in the guilty set
        mask = ~sales_person["sales_id"].isin(red_sales_ids)

        log.debug("Sales people passing the ~isin filter (no RED orders): %d / %d",
                  mask.sum(), len(sales_person))

        result = sales_person[mask][["name"]].reset_index(drop=True)

        log.info("Output shape: %d rows x %d cols", *result.shape)
        log.info("Result:\n%s\n", result.to_string(index=False))
        return result

    @staticmethod
    def verify(result: pd.DataFrame) -> None:
        res = result.sort_values("name").reset_index(drop=True)

        expected = pd.DataFrame({"name": ["Charlie", "Diana", "Eve"]})

        pd.testing.assert_frame_equal(res, expected, check_dtype=False)
        log.info("✅  All assertions passed – output matches expected values.")


if __name__ == "__main__":
    solver = SalesPerson()

    sales_person, company, orders = solver.make_sample_data()
    result = solver.sales_person(sales_person, company, orders)
    solver.verify(result)

    # ── Demonstrate why ~ is not the same as != ──────────────────────
    log.info("--- Contrast: != vs ~isin() on a set ---")
    ids = pd.Series([1, 2, 1, 3])
    guilty = [1, 2]
    log.info("ids != 1          → %s  (only excludes the literal value 1)",
             (ids != 1).tolist())
    log.info("~ids.isin([1,2])  → %s  (excludes the whole set)",
             (~ids.isin(guilty)).tolist())

    # ── Edge case: RED does not exist in company ──────────────────────
    log.info("--- Edge case: RED not in company table ---")
    no_red = company[company["name"] != "RED"]
    result2 = solver.sales_person(sales_person, no_red, orders)
    log.info("No RED company → all %d sales people returned", len(result2))

    # ── Edge case: no orders at all ───────────────────────────────────
    log.info("--- Edge case: empty orders table ---")
    empty_orders = pd.DataFrame(columns=["order_id", "order_date", "com_id", "sales_id", "amount"])
    result3 = solver.sales_person(sales_person, company, empty_orders)
    log.info("No orders → all %d sales people returned", len(result3))