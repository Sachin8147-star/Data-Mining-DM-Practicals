students = [
    {"id": 1, "name": "Shyam", "marks": 85, "attendance": 92},
    {"id": 2, "name": "Priya", "marks": 72, "attendance": 85},
    {"id": 3, "name": "mukul", "marks": 55, "attendance": 68},
    {"id": 4, "name": "Nehal", "marks": 91, "attendance": 95},
    {"id": 5, "name": "Ravin", "marks": 45, "attendance": 60},
    {"id": 6, "name": "Sneha", "marks": 78, "attendance": 88},
    {"id": 7, "name": "Karam", "marks": 62, "attendance": 75},
    {"id": 8, "name": "Akash", "marks": 39, "attendance": 55}
]


# Original Data
print("----- STUDENT DATA -----")
print("ID", "Name", "Marks", "Attendance")

for s in students:
    print(s["id"], s["name"], s["marks"], s["attendance"])


# 1. NORMALIZATION - Min-Max
all_marks = [s["marks"] for s in students]

low = min(all_marks)
high = max(all_marks)

print("\n----- NORMALIZATION -----")

for s in students:
    value = (s["marks"] - low) / (high - low)
    print(s["name"], "=>", round(value, 2))


# 2. STANDARDIZATION - Z-Score
mean = sum(all_marks) / len(all_marks)

variance = sum((x - mean) ** 2 for x in all_marks) / len(all_marks)
std = variance ** 0.5

print("\n----- STANDARDIZATION -----")

for s in students:
    value = (s["marks"] - mean) / std
    print(s["name"], "=>", round(value, 2))


# 3. TRANSFORMATION
print("\n----- TRANSFORMATION -----")

for s in students:
    percentage = (s["marks"] / 100) * 100
    print(s["name"], "=>", percentage, "%")


# 4. AGGREGATION
sum_marks = 0

for s in students:
    sum_marks += s["marks"]

avg_marks = sum_marks / len(students)

print("\n----- AGGREGATION -----")
print("Total Marks:", sum_marks)
print("Average Marks:", round(avg_marks, 2))


# 5. DISCRETIZATION
print("\n----- DISCRETIZATION -----")

for s in students:
    score = s["marks"]

    if score < 50:
        level = "Poor"
    elif score < 75:
        level = "Average"
    else:
        level = "Good"

    print(s["name"], "=>", level)


# 6. BINARIZATION
print("\n----- BINARIZATION -----")

for s in students:
    if s["marks"] >= 50:
        status = 1
    else:
        status = 0

    print(s["name"], "=>", status)


# 7. SAMPLING
print("\n----- SAMPLING -----")
print("Alternate Records:")

for i in range(0, len(students), 2):
    print(students[i])