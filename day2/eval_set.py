"""
eval_set.py — the evaluation set: natural-language questions paired with
gold-standard SQL. Covers a spread of query types so the eval measures
real capability, not one narrow pattern.

Query types covered:
  - simple select / filter
  - aggregation (COUNT, AVG, MAX)
  - GROUP BY
  - JOIN (single and multi)
  - ORDER BY / LIMIT
  - subquery / HAVING
"""

EVAL_SET = [
    {
        "id": "q01",
        "question": "List the names of all students.",
        "gold_sql": "SELECT name FROM students;",
        "type": "simple",
    },
    {
        "id": "q02",
        "question": "How many students are there?",
        "gold_sql": "SELECT COUNT(*) FROM students;",
        "type": "aggregation",
    },
    {
        "id": "q03",
        "question": "What is the average GPA of all students?",
        "gold_sql": "SELECT AVG(gpa) FROM students;",
        "type": "aggregation",
    },
    {
        "id": "q04",
        "question": "List the names of students with a GPA above 3.5.",
        "gold_sql": "SELECT name FROM students WHERE gpa > 3.5;",
        "type": "filter",
    },
    {
        "id": "q05",
        "question": "How many students are in each department? Show the department name and the count.",
        "gold_sql": "SELECT d.name, COUNT(s.student_id) FROM departments d LEFT JOIN students s ON d.dept_id = s.dept_id GROUP BY d.dept_id;",
        "type": "group_by_join",
    },
    {
        "id": "q06",
        "question": "Which department has the highest budget? Return its name.",
        "gold_sql": "SELECT name FROM departments ORDER BY budget DESC LIMIT 1;",
        "type": "order_limit",
    },
    {
        "id": "q07",
        "question": "List the titles of courses offered by the Computer Science department.",
        "gold_sql": "SELECT c.title FROM courses c JOIN departments d ON c.dept_id = d.dept_id WHERE d.name = 'Computer Science';",
        "type": "join_filter",
    },
    {
        "id": "q08",
        "question": "What is the name of the student with the highest GPA?",
        "gold_sql": "SELECT name FROM students ORDER BY gpa DESC LIMIT 1;",
        "type": "order_limit",
    },
    {
        "id": "q09",
        "question": "How many courses is each student enrolled in? Show student name and count.",
        "gold_sql": "SELECT s.name, COUNT(e.course_id) FROM students s JOIN enrollments e ON s.student_id = e.student_id GROUP BY s.student_id;",
        "type": "group_by_join",
    },
    {
        "id": "q10",
        "question": "List the names of departments that have more than 2 students.",
        "gold_sql": "SELECT d.name FROM departments d JOIN students s ON d.dept_id = s.dept_id GROUP BY d.dept_id HAVING COUNT(s.student_id) > 2;",
        "type": "having",
    },
    {
        "id": "q11",
        "question": "What is the average grade in the course titled 'Algorithms'?",
        "gold_sql": "SELECT AVG(e.grade) FROM enrollments e JOIN courses c ON e.course_id = c.course_id WHERE c.title = 'Algorithms';",
        "type": "join_aggregation",
    },
    {
        "id": "q12",
        "question": "List the names of students enrolled in the 'Databases' course.",
        "gold_sql": "SELECT s.name FROM students s JOIN enrollments e ON s.student_id = e.student_id JOIN courses c ON e.course_id = c.course_id WHERE c.title = 'Databases';",
        "type": "multi_join",
    },
    {
        "id": "q13",
        "question": "How many students enrolled in each year? Show the year and the count.",
        "gold_sql": "SELECT enrolled_year, COUNT(*) FROM students GROUP BY enrolled_year;",
        "type": "group_by",
    },
    {
        "id": "q14",
        "question": "Which student has taken the most courses? Return their name.",
        "gold_sql": "SELECT s.name FROM students s JOIN enrollments e ON s.student_id = e.student_id GROUP BY s.student_id ORDER BY COUNT(e.course_id) DESC LIMIT 1;",
        "type": "group_order_limit",
    },
    {
        "id": "q15",
        "question": "List the names of students who are not enrolled in any course.",
        "gold_sql": "SELECT name FROM students WHERE student_id NOT IN (SELECT student_id FROM enrollments);",
        "type": "subquery",
    },
]

if __name__ == "__main__":
    from collections import Counter
    types = Counter(q["type"] for q in EVAL_SET)
    print(f"{len(EVAL_SET)} eval questions")
    for t, n in types.most_common():
        print(f"  {t}: {n}")
