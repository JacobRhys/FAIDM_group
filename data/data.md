# Open University Learning Analytics Dataset (OULAD)

## Overview

The **Open University Learning Analytics Dataset (OULAD)** is an openly available dataset released by the Knowledge Media Institute at The Open University. It is designed to support research into learning analytics, educational data mining, and student performance modelling.

The dataset contains anonymised information about:

- Courses and their presentations
- Students and demographic attributes  
- Assessments and grades  
- Student interaction with online learning materials (VLE)

All data is linked through a small number of shared identifiers, making it suitable for relational analysis and machine learning workflows.

---

## Dataset Structure

The dataset is provided as a collection of CSV files. Each file represents a logical entity, and relationships are defined using shared keys such as:

- `code_module`
- `code_presentation`
- `id_student`
- `id_assessment`
- `id_site`

---

## File Descriptions

### `courses.csv`

Defines the modules and their presentations.

| Column | Description |
|------|-------------|
| code_module | Module identifier |
| code_presentation | Presentation identifier |
| length | Duration of the course in days |

This table defines the academic context used across the dataset.

---

### `assessments.csv`

Contains metadata for all assessments in each module.

| Column | Description |
|------|-------------|
| id_assessment | Unique assessment ID |
| code_module | Module code |
| code_presentation | Presentation code |
| assessment_type | TMA, CMA, or Exam |
| date | Days from start of presentation |
| weight | Contribution to final grade |

Each assessment belongs to exactly one module presentation.

---

### `vle.csv`

Describes learning materials available in the Virtual Learning Environment.

| Column | Description |
|------|-------------|
| id_site | Learning material ID |
| code_module | Module code |
| code_presentation | Presentation code |
| activity_type | Type of resource |
| week_from | First week available |
| week_to | Last week available |

This table defines what content students can interact with.

---

### `studentInfo.csv`

Contains demographic and outcome information for students.

| Column | Description |
|------|-------------|
| id_student | Student identifier |
| code_module | Module code |
| code_presentation | Presentation code |
| gender | Student gender |
| region | Geographic region |
| highest_education | Education level |
| age_band | Age group |
| studied_credits | Credit load |
| disability | Disability indicator |
| final_result | Pass, Fail, Withdrawn, etc. |

This is the central table for student-level analysis.

---

### `studentRegistration.csv`

Tracks registration and withdrawal dates.

| Column | Description |
||------|-------------|
| id_student | Student identifier |
| code_module | Module code |
| code_presentation | Presentation code |
| date_registration | Days since start |
| date_unregistration | Days since start (if withdrawn) |

Useful for studying engagement duration and dropout behaviour.

---

### `studentAssessment.csv`

Records student submissions and results.

| Column | Description |
|------|-------------|
| id_student | Student identifier |
| id_assessment | Assessment identifier |
| date_submitted | Days since start |
| is_banked | Credit carried over |
| score | Assessment score |

Links students to their assessment outcomes.

---

### `studentVle.csv`

Contains interaction data between students and the VLE.

| Column | Description |
|------|-------------|
| id_student | Student identifier |
| code_module | Module code |
| code_presentation | Presentation code |
| id_site | Learning resource |
| date | Day of interaction |
| sum_click | Number of clicks |

This is the largest table and represents behavioural activity.

---

## Relationships Between Tables

courses
│
├── assessments ── studentAssessment
│
├── vle ────────── studentVle
│
└── studentInfo ── studentRegistration

### Key Relationships

- `code_module` + `code_presentation`
  - Link courses, assessments, VLE resources, and students

- `id_student`
  - Links student information, registrations, assessments, and VLE activity

- `id_assessment`
  - Connects assessment definitions to student scores

- `id_site`
  - Links learning materials to interaction logs

---

## Common Analytical Uses

The dataset supports:

- Student performance prediction  
- Dropout and retention analysis  
- Engagement modelling using clickstream data  
- Assessment effectiveness studies  
- Behavioural clustering and pattern mining  
- Time-series analysis of learning activity  

---

## License and Citation

**License:** Creative Commons Attribution 4.0 (CC BY 4.0)

**Recommended citation:**

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017).  
*Open University Learning Analytics Dataset*. Scientific Data, 4, 170171.

---

## Summary

The OULAD dataset provides a well-structured, relational view of student learning behaviour. By combining demographic data, assessment results, and fine-grained activity logs, it enables comprehensive educational data analysis and predictive modelling.
