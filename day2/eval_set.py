"""
eval_set.py — ADVERSARIAL text-to-SQL eval questions.

Each question targets a specific failure mode of base LLMs on SQL.
The 'trap' field documents what a naive model typically gets wrong.
"""

EVAL_SET = [
    {
        "id": "h01",
        "question": "List every department name along with how many employees it has. Include departments that have no employees.",
        "gold_sql": """SELECT d.dept_nm, COUNT(e.emp_id) AS n
                       FROM depts d LEFT JOIN employees e ON d.dept_cd = e.dept_cd
                       GROUP BY d.dept_cd;""",
        "type": "left_join",
        "trap": "inner JOIN silently drops the HR department (zero employees)",
    },
    {
        "id": "h02",
        "question": "What is the average salary across all employees?",
        "gold_sql": "SELECT AVG(sal) FROM employees;",
        "type": "null_aggregation",
        "trap": "AVG ignores NULL salaries; naive SUM/COUNT(*) divides by 7 not 6",
    },
    {
        "id": "h03",
        "question": "How many employees have a recorded salary?",
        "gold_sql": "SELECT COUNT(sal) FROM employees;",
        "type": "null_aggregation",
        "trap": "COUNT(*) returns 7; COUNT(sal) correctly returns 6",
    },
    {
        "id": "h04",
        "question": "Return the names of all employees who earn the highest salary.",
        "gold_sql": """SELECT emp_nm FROM employees
                       WHERE sal = (SELECT MAX(sal) FROM employees);""",
        "type": "subquery_max",
        "trap": "TWO employees tie at the max (200000). 'ORDER BY sal DESC LIMIT 1' returns only one -> wrong.",
    },
    {
        "id": "h05",
        "question": "List the names of employees who earn more than the average salary of their own department.",
        "gold_sql": """SELECT e.emp_nm FROM employees e
                       WHERE e.sal > (SELECT AVG(e2.sal) FROM employees e2
                                      WHERE e2.dept_cd = e.dept_cd);""",
        "type": "correlated_subquery",
        "trap": "requires a correlated subquery; naive versions compare to global average",
    },
    {
        "id": "h06",
        "question": "List each employee's name together with their manager's name. Include employees who have no manager.",
        "gold_sql": """SELECT e.emp_nm, m.emp_nm FROM employees e
                       LEFT JOIN employees m ON e.mgr_id = m.emp_id;""",
        "type": "self_join_null",
        "trap": "self-join; inner join drops the CEO whose mgr_id is NULL",
    },
    {
        "id": "h07",
        "question": "Which employees are not assigned to any project? Return their names.",
        "gold_sql": """SELECT emp_nm FROM employees
                       WHERE emp_id NOT IN (SELECT emp_id FROM assignments);""",
        "type": "anti_join",
        "trap": "NOT IN / NOT EXISTS / LEFT JOIN ... IS NULL all valid; must find Idle Ian",
    },
    {
        "id": "h08",
        "question": "List each project name and the total hours logged against it. Include projects with no assignments, showing zero.",
        "gold_sql": """SELECT p.proj_nm, COALESCE(SUM(a.hrs), 0) AS total_hrs
                       FROM projects p LEFT JOIN assignments a ON p.proj_id = a.proj_id
                       GROUP BY p.proj_id;""",
        "type": "left_join_coalesce",
        "trap": "SUM over no rows returns NULL, not 0; and inner join drops Moonshot",
    },
    {
        "id": "h09",
        "question": "For each department, show the department name and its region name. Include departments with no region assigned.",
        "gold_sql": """SELECT d.dept_nm, r.region_nm FROM depts d
                       LEFT JOIN regions r ON d.region_cd = r.region_cd;""",
        "type": "left_join_null_fk",
        "trap": "RND has NULL region_cd; inner join drops it",
    },
    {
        "id": "h10",
        "question": "Show each department name and its 2025 budget amount. Include departments that have no budget row.",
        "gold_sql": """SELECT d.dept_nm, b.amt FROM depts d
                       LEFT JOIN budgets b ON d.dept_cd = b.dept_cd AND b.fy = 2025;""",
        "type": "left_join_with_condition",
        "trap": "putting b.fy=2025 in WHERE instead of ON turns the LEFT JOIN into an inner join",
    },
    {
        "id": "h11",
        "question": "Rank employees within each department by salary from highest to lowest, showing employee name, department code, and their rank.",
        "gold_sql": """SELECT emp_nm, dept_cd,
                              RANK() OVER (PARTITION BY dept_cd ORDER BY sal DESC) AS rnk
                       FROM employees;""",
        "type": "window_function",
        "trap": "requires a window function; also NULL salary sorts unpredictably",
    },
    {
        "id": "h12",
        "question": "For each department, return the name of the employee with the highest salary in that department.",
        "gold_sql": """SELECT e.dept_cd, e.emp_nm FROM employees e
                       WHERE e.sal = (SELECT MAX(e2.sal) FROM employees e2
                                      WHERE e2.dept_cd = e.dept_cd);""",
        "type": "correlated_max",
        "trap": "GROUP BY dept_cd with MAX(sal) does NOT give you the right emp_nm",
    },
    {
        "id": "h13",
        "question": "Count how many employees each manager directly manages. Show the manager's name and the count.",
        "gold_sql": """SELECT m.emp_nm, COUNT(e.emp_id) FROM employees m
                       JOIN employees e ON e.mgr_id = m.emp_id
                       GROUP BY m.emp_id;""",
        "type": "self_join_aggregate",
        "trap": "self-join direction is easy to invert",
    },
    {
        "id": "h14",
        "question": "List the names of employees hired before 2020 who work in a department located in North America.",
        "gold_sql": """SELECT e.emp_nm FROM employees e
                       JOIN depts d ON e.dept_cd = d.dept_cd
                       JOIN regions r ON d.region_cd = r.region_cd
                       WHERE r.region_nm = 'North America' AND e.hire_dt < '2020-01-01';""",
        "type": "multi_hop_join_date",
        "trap": "three-table hop plus string date comparison",
    },
    {
        "id": "h15",
        "question": "Which departments have more than one employee earning at least 95000? Return the department name and that count.",
        "gold_sql": """SELECT d.dept_nm, COUNT(*) FROM employees e
                       JOIN depts d ON e.dept_cd = d.dept_cd
                       WHERE e.sal >= 95000
                       GROUP BY d.dept_cd HAVING COUNT(*) > 1;""",
        "type": "having_filtered",
        "trap": "filter must apply before aggregation; HAVING vs WHERE confusion",
    },
    {
        "id": "h16",
        "question": "Return the total hours logged by employees in the Engineering department.",
        "gold_sql": """SELECT SUM(a.hrs) FROM assignments a
                       JOIN employees e ON a.emp_id = e.emp_id
                       WHERE e.dept_cd = 'ENG';""",
        "type": "join_filter_aggregate",
        "trap": "must join through employees, not projects (proj dept != employee dept)",
    },
]

if __name__ == "__main__":
    from collections import Counter
    print(f"{len(EVAL_SET)} hard eval questions\n")
    for q in EVAL_SET:
        print(f"  {q['id']} [{q['type']}]")
        print(f"      trap: {q['trap']}")
