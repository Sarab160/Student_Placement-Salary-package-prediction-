import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
df=pd.read_csv("student.csv")

# print(df.head(5))
# print(df.columns)
# print(df.info())
# print(df.describe())

# sns.pairplot(df)
# plt.show()
df["volunteer_experience"] = df["volunteer_experience"].map({
    "No": 0,
    "Yes": 1
})

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
d_le=df["gender"]
encode_g=le.fit_transform(df["gender"])
le_encode=pd.DataFrame(encode_g,columns=["gender"])

X_1=pd.concat([x,le_encode],axis=1)

#=========== onhot =====
ohe=OneHotEncoder(sparse_output=False,drop="first")
ohe_col=df[["branch"]]
t=ohe.fit_transform(ohe_col)
t_data=pd.DataFrame(t,columns=ohe.get_feature_names_out(["branch"]))

X_2=pd.concat([X_1,t_data],axis=1)

ode=OrdinalEncoder()
ode_encode=ode.fit_transform(df[["college_tier"]])
o_data=pd.DataFrame(ode_encode,columns=["college_tier"])

X=pd.concat([X_2,o_data],axis=1)

# ss=StandardScaler()
# X_final=pd.DataFrame(data=ss.fit_transform(X),columns=x.columns)

x_train,x_test,y_train_,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

# print(X.shape)

ann=Sequential()

