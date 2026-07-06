"""
build_db.py — creates a small, realistic multi-table SQLite database
used as the target for text-to-SQL evaluation.

Domain-neutral (a university/course domain, Spider-style) so the focus
stays on the NL->SQL technique, not any vertical.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "eval.db")

SCHEMA = """
CREATE TABLE departments (
    dept_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    budget      INTEGER NOT NULL
);

CREATE TABLE students (
    student_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    dept_id     INTEGER,
    gpa         REAL,
    enrolled_year INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

CREATE TABLE courses (
    course_id   INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    dept_id     INTEGER,
    credits     INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

CREATE TABLE enrollments (
    student_id  INTEGER,
    course_id   INTEGER,
    grade       REAL,
    term        TEXT,
    PRIMARY KEY (student_id, course_id, term),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
"""

DEPARTMENTS = [
    (1, "Computer Science", 500000),
    (2, "Mathematics", 300000),
    (3, "Physics", 350000),
    (4, "History", 200000),
]

STUDENTS = [
    (1, "Alice Chen", 1, 3.9, 2022),
    (2, "Bob Kumar", 1, 3.2, 2021),
    (3, "Carla Diaz", 2, 3.7, 2022),
    (4, "Deepak Rao", 3, 3.5, 2020),
    (5, "Eva Novak", 1, 2.8, 2023),
    (6, "Frank Li", 4, 3.4, 2021),
    (7, "Grace Park", 2, 3.95, 2022),
    (8, "Hassan Ali", 3, 3.1, 2023),
]

COURSES = [
    (101, "Intro to Programming", 1, 4),
    (102, "Algorithms", 1, 3),
    (103, "Linear Algebra", 2, 3),
    (104, "Quantum Mechanics", 3, 4),
    (105, "World History", 4, 3),
    (106, "Databases", 1, 3),
]

ENROLLMENTS = [
    (1, 101, 4.0, "Fall2022"),
    (1, 102, 3.7, "Spring2023"),
    (1, 106, 4.0, "Fall2023"),
    (2, 101, 3.0, "Fall2021"),
    (2, 102, 2.7, "Spring2022"),
    (3, 103, 3.9, "Fall2022"),
    (4, 104, 3.3, "Fall2020"),
    (5, 101, 2.5, "Fall2023"),
    (6, 105, 3.6, "Fall2021"),
    (7, 103, 4.0, "Fall2022"),
    (8, 104, 3.0, "Spring2023"),
    (1, 103, 3.8, "Fall2022"),
]


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    cur.executemany("INSERT INTO departments VALUES (?,?,?)", DEPARTMENTS)
    cur.executemany("INSERT INTO students VALUES (?,?,?,?,?)", STUDENTS)
    cur.executemany("INSERT INTO courses VALUES (?,?,?,?)", COURSES)
    cur.executemany("INSERT INTO enrollments VALUES (?,?,?,?)", ENROLLMENTS)
    conn.commit()
    conn.close()
    print(f"Built {DB_PATH}")


def schema_text():
    """Human/LLM-readable schema description injected into the prompt."""
    return """Tables:
departments(dept_id, name, budget)
students(student_id, name, dept_id, gpa, enrolled_year)
courses(course_id, title, dept_id, credits)
enrollments(student_id, course_id, grade, term)

Foreign keys:
students.dept_id -> departments.dept_id
courses.dept_id -> departments.dept_id
enrollments.student_id -> students.student_id
enrollments.course_id -> courses.course_id"""


if __name__ == "__main__":
    build()
