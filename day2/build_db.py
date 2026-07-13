"""
build_db.py — a deliberately ADVERSARIAL text-to-SQL evaluation database.

Two design goals, learned the hard way when base Qwen scored 100% on the
first (too-easy) version of this eval:

  1. HARD SCHEMA
     - abbreviated / non-obvious column names (emp_id, mgr_id, dept_cd, sal)
     - a self-referential FK (employees.mgr_id -> employees.emp_id)
     - a many-to-many with attributes (assignments)
     - NULLable columns that matter
     - a lookup table requiring a multi-hop join

  2. DISCRIMINATING DATA
     The fixture is built so that WRONG-BUT-PLAUSIBLE queries give DIFFERENT
     answers than correct ones. Specifically:
       - a department with ZERO employees  -> LEFT JOIN != INNER JOIN
       - an employee with NO assignments   -> exposes join-type errors
       - a project with NO assignments     -> same
       - NULL salaries                     -> AVG/COUNT semantics matter
       - NULL mgr_id (the CEO)             -> self-join must handle it
       - TIES in salary                    -> naive "ORDER BY .. LIMIT 1" is wrong
       - an employee in a dept whose
         budget row is missing             -> multi-hop join must be careful
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "eval.db")

SCHEMA = """
CREATE TABLE depts (
    dept_cd     TEXT PRIMARY KEY,     -- 'ENG', 'SLS', ...
    dept_nm     TEXT NOT NULL,
    region_cd   TEXT                  -- FK to regions; NULLable
);

CREATE TABLE regions (
    region_cd   TEXT PRIMARY KEY,
    region_nm   TEXT NOT NULL
);

CREATE TABLE budgets (
    dept_cd     TEXT PRIMARY KEY,
    fy          INTEGER NOT NULL,
    amt         INTEGER
);

CREATE TABLE employees (
    emp_id      INTEGER PRIMARY KEY,
    emp_nm      TEXT NOT NULL,
    dept_cd     TEXT,                 -- FK depts
    mgr_id      INTEGER,              -- self-FK; NULL for the CEO
    sal         INTEGER,              -- NULLable (contractors)
    hire_dt     TEXT                  -- ISO date string
);

CREATE TABLE projects (
    proj_id     INTEGER PRIMARY KEY,
    proj_nm     TEXT NOT NULL,
    dept_cd     TEXT
);

CREATE TABLE assignments (
    emp_id      INTEGER,
    proj_id     INTEGER,
    hrs         INTEGER,
    PRIMARY KEY (emp_id, proj_id)
);
"""

REGIONS = [
    ("NA", "North America"),
    ("EU", "Europe"),
    ("APAC", "Asia Pacific"),
]

# NOTE: 'HR' dept has NO employees  -> LEFT vs INNER JOIN discriminator
# NOTE: 'RND' has a NULL region_cd  -> multi-hop join must handle NULL
DEPTS = [
    ("ENG", "Engineering", "NA"),
    ("SLS", "Sales",       "EU"),
    ("HR",  "Human Resources", "NA"),   # zero employees
    ("RND", "Research",    None),       # NULL region
]

# NOTE: no budget row for 'RND'  -> multi-hop join discriminator
BUDGETS = [
    ("ENG", 2025, 900000),
    ("SLS", 2025, 400000),
    ("HR",  2025, 150000),
]

# emp 1 = CEO (mgr_id NULL)
# emp 6 has NULL salary (contractor)  -> AVG/COUNT semantics
# emps 3 and 4 TIE at 95000           -> naive ORDER BY..LIMIT 1 is wrong
# emp 7 has NO assignments            -> LEFT vs INNER JOIN discriminator
# emp 1 and emp 2 TIE at the MAXIMUM salary (200000)
#   -> "ORDER BY sal DESC LIMIT 1" returns ONE row; the correct
#      "WHERE sal = (SELECT MAX(sal))" returns TWO. This is the trap.
EMPLOYEES = [
    (1, "Ada Lovelace",   "ENG", None, 200000, "2015-01-10"),  # CEO, tie for max
    (2, "Grace Hopper",   "ENG", 1,    200000, "2017-03-22"),  # tie for max
    (3, "Alan Turing",    "ENG", 2,     95000, "2019-06-01"),
    (4, "Katherine J",    "ENG", 2,     95000, "2019-06-01"),
    (5, "Linus T",        "SLS", 1,     88000, "2020-09-15"),
    (6, "Contractor Bob", "SLS", 5,      None, "2023-01-05"),  # NULL salary
    (7, "Idle Ian",       "RND", 1,     70000, "2021-11-30"),  # no assignments
]

# proj 30 has NO assignments -> discriminator
PROJECTS = [
    (10, "Compiler",   "ENG"),
    (20, "CRM Rollout","SLS"),
    (30, "Moonshot",   "RND"),   # nobody assigned
]

ASSIGNMENTS = [
    (1, 10, 100),
    (2, 10, 220),
    (3, 10, 300),
    (4, 10, 150),
    (5, 20, 200),
    (6, 20,  40),
    # emp 7 deliberately absent
]


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    cur.executemany("INSERT INTO regions VALUES (?,?)", REGIONS)
    cur.executemany("INSERT INTO depts VALUES (?,?,?)", DEPTS)
    cur.executemany("INSERT INTO budgets VALUES (?,?,?)", BUDGETS)
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", EMPLOYEES)
    cur.executemany("INSERT INTO projects VALUES (?,?,?)", PROJECTS)
    cur.executemany("INSERT INTO assignments VALUES (?,?,?)", ASSIGNMENTS)
    conn.commit()
    conn.close()
    print(f"Built {DB_PATH}")


def schema_text():
    """Schema description injected into the model prompt. Deliberately terse —
    real schemas don't come with friendly explanations."""
    return """Tables:
regions(region_cd, region_nm)
depts(dept_cd, dept_nm, region_cd)
budgets(dept_cd, fy, amt)
employees(emp_id, emp_nm, dept_cd, mgr_id, sal, hire_dt)
projects(proj_id, proj_nm, dept_cd)
assignments(emp_id, proj_id, hrs)

Foreign keys:
depts.region_cd -> regions.region_cd
budgets.dept_cd -> depts.dept_cd
employees.dept_cd -> depts.dept_cd
employees.mgr_id -> employees.emp_id
projects.dept_cd -> depts.dept_cd
assignments.emp_id -> employees.emp_id
assignments.proj_id -> projects.proj_id

Notes: sal and mgr_id and region_cd may be NULL. hire_dt is an ISO date string."""


if __name__ == "__main__":
    build()
