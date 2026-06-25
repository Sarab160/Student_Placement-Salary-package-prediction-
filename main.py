import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
df=pd.read_csv("student.csv")

# print(df.head(5))
# print(df.columns)
# print(df.info())
# print(df.describe())

# sns.pairplot(df)
# plt.show()
x=df[["age", 'cgpa',
        'internships_count', 'projects_count', 'certifications_count',
        'coding_skill_score', 'aptitude_score', 'communication_skill_score',
        'logical_reasoning_score', 'hackathons_participated', 'github_repos',
        'linkedin_connections', 'mock_interview_score', 'attendance_percentage',
        'backlogs', 'extracurricular_score', 'leadership_score',
        'volunteer_experience', 'sleep_hours', 'study_hours_per_day',]]

y=df["salary_package_lpa"]

print(df["branch"].unique())

le=LabelEncoder()
encode_g=le.fit_transform(df["gender"])
# 'student_id', 'age', 'gender', 'cgpa', 'branch', 'college_tier',
#        'internships_count', 'projects_count', 'certifications_count',
#        'coding_skill_score', 'aptitude_score', 'communication_skill_score',
#        'logical_reasoning_score', 'hackathons_participated', 'github_repos',
#        'linkedin_connections', 'mock_interview_score', 'attendance_percentage',
#        'backlogs', 'extracurricular_score', 'leadership_score',
#        'volunteer_experience', 'sleep_hours', 'study_hours_per_day',
#        'placement_status', 'salary_package_lpa']