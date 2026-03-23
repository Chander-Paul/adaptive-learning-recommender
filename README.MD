# Adaptive Learning Resource Recommender App

This is an adaptive Learning Recommnder Appplication utilizing ML and data science to recommend resources to users.

Utilizs affect tracking to recommend resources during study and recommendation/elo mastery to recommend further learning.




https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style/data
source for personalised data.
student performance dataset headings
Column Descriptions
StudyHours – Number of study hours per week.
Attendance – Percentage of classes attended.
Resources – Availability and use of academic resources (e.g., library, notes).
Extracurricular – Participation in extracurricular activities (Yes/No).
Motivation – Self-reported motivation level (numeric scale).
Internet – Access to the internet for study purposes (Yes/No).
Gender – Student’s gender (Male/Female).
Age – Age of student (18–30 years).
LearningStyle – Preferred learning style (e.g., Visual, Auditory, Kinesthetic, Reading/Writing).
OnlineCourses – Participation in online courses (Yes/No).
Discussions – Engagement in study group discussions or forums.
AssignmentCompletion – Rate of completing assignments on time (numeric scale).
ExamScore – Score obtained in the main exam.
EduTech – Usage of educational technology tools/platforms.
StressLevel – Self-reported stress level (numeric scale).
FinalGrade – Final course grade (target variable for prediction).


Rune file instructions with Make

``` make reset-db``` Reset database initialisation

```make run-api``` Run api server with connection to database.

```./.venv/bin/python scripts/populate_content_database.py --map-resource-types``` Seed content using the README resource-type mapping.

# Resources.

### Types of Resources

Survey     -> resource
Library    -> resource
Resource
Tutorial   ->resource
resources   -> resource
naclo       -> assignment
course      -> resource
corpus      -> resource
lecture     -> resource
paper       -> resources
Course      -> resources
corpora     -> resources
Corpora     -> resources
Surveys     -> resources
Libraries   -> resources
Resources   -> resources
Tutorials   -> resources
NACLO Problems  -> assignments
Lectures        -> resources
Papers          -> resources
Other           -> resources
Books           -> resources
book            -> resources

1



