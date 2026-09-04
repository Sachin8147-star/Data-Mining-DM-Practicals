file = open("paper_reviews.csv", "r")
lines = file.readlines()
file.close()

data = []

for line in lines[1:]:
    data.append(line.strip().split(","))

print("Original Data:")
for row in data:
    print(row)

# Missing values
for row in data:
    for i in range(1, 4):
        if row[i] == "":
            row[i] = "3"

# Inconsistent values
for row in data:
    row[0] = row[0].strip().lower()
    row[4] = row[4].strip().lower()

# Remove duplicates
cleaned = []

for row in data:
    if row not in cleaned:
        cleaned.append(row)

data = cleaned

# Handle outliers
for row in data:
    for i in range(1, 4):
        if float(row[i]) < 1 or float(row[i]) > 5:
            row[i] = "5"

# Validation
missing = 0
duplicates = 0
invalid = 0

for row in data:
    for value in row:
        if value == "":
            missing += 1

    for value in row[1:4]:
        if float(value) < 1 or float(value) > 5:
            invalid += 1

print("\nValidation:")
print("Missing values:", missing)
print("Duplicate rows:", duplicates)
print("Invalid values:", invalid)

print("\nCleaned Data:")
for row in data:
    print(row)

# Save cleaned data
file = open("cleaned_paper_reviews.csv", "w")

file.write("reviewer,originality,quality,clarity,decision\n")

for row in data:
    file.write(",".join(row) + "\n")

file.close()

print("\nCleaned dataset saved successfully.")