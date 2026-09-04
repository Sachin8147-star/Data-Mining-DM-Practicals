import pandas as pd
import random

names=["Aarav","Vivaan","Aditya","Vihaan","Arjun","Reyansh","Muhammad","Sai","Ayaan","Krishna",
       "Ishaan","Shaurya","Atharv","Rudra","Ansh","Advik","Dhruv","Kabir","Aryan","Aarush","Raghav","Shivansh","Om","Anay",
       "Pranav","Yuvraj","Aarav","Vivaan","Aditya"]

surnames=["Sharma","Verma","Gupta","Mehta","Jain","Kapoor","Malhotra","Chopra","Bhatia","Agarwal",
          "Singh","Kumar","Reddy","Patel","Nair","Iyer","Rao","Chatterjee","Mukherjee","Das","Bose","Ghosh","Sen","Roy","Dutta",
          "Saxena","Choudhary","Trivedi","Joshi","Shah"]

data = []

for i in range(1,201):
    row={
        "ID":i,
        "Name":random.choice(names),
        "Surname":random.choice(surnames),
        "Maths":random.randint(0,50),
        "Science":random.randint(0,50),
        "English":random.randint(0,50),
        "History":random.randint(0,50),
        "Geography":random.randint(0,50)
    }
    data.append(row)

dataframe=pd.DataFrame(data)

dataframe.to_csv("student_data.csv",index=False)
print("CSV file 'student_data.csv' has been created successfully.")

#(a) Average marks of all subjects
print("\nAverage marks of all subjects:")
print(dataframe[["Maths","Science","English","History","Geography"]].mean())

#(b) overall Topper
dataframe["Total"] = dataframe[["Maths","Science","English","History","Geography"]].sum(axis=1)
topper = dataframe.loc[dataframe["Total"].idxmax()]
print("\nOverall Topper:")
print(topper[["ID","Name","Surname","Total"]])

#(c) Subject-wise Toppers
print("\nSubject-wise Toppers:")
subjects = ["Maths","Science","English","History","Geography"]

for subject in subjects:
    topper = dataframe.loc[dataframe[subject].idxmax()]
    print(f"{subject} Topper: ID: {topper['ID']}, Name: {topper['Name']}, Surname: {topper['Surname']}, Marks: {topper[subject]}")

#(d) Students scoring less than 15 in any subject
low_marks=dataframe[(dataframe["Maths"] < 15) | (dataframe["Science"] < 15) | (dataframe["English"] < 15) | (dataframe["History"] < 15) | (dataframe["Geography"] < 15)]
print("\nStudents scoring less than 15 in any subject:")
print(low_marks[["ID","Name","Surname"]])

low_marks.to_csv("low_marks_students.csv",index=False)
