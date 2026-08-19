import csv
import random

names = [
    "Muhammad Ahmed",
    "Abdullah Khan",
    "Ali Hassan",
    "Usman Ali",
    "Hamza Ahmed",
    "Bilal Hussain",
    "Hassan Raza",
    "Hussain Ali",
    "Omar Farooq",
    "Talha Saeed",
    "Zain Abbas",
    "Saad Ahmed",
    "Ayaan Malik",
    "Daniyal Khan",
    "Huzaifa Ahmed",
    "Abdul Rehman",
    "Ahmad Raza",
    "Mohammad Usman",
    "Arham Shah",
    "Rayyan Ali",
    "Ibrahim Khan",
    "Yusuf Ahmed",
    "Ismail Hassan",
    "Musa Khan",
    "Suleman Ahmed",
    "Hamza Farooq",
    "Fahad Ali",
    "Shahzaib Khan",
    "Taha Hussain",
    "Waleed Ahmed",
    "Adeel Raza",
    "Asad Khan",
    "Imran Ali",
    "Faisal Ahmed",
    "Saifullah Khan",
    "Abdul Basit",
    "Abdul Hadi",
    "Abdul Wahab",
    "Junaid Ahmed",
    "Kamran Ali",
    "Noman Khan",
    "Rizwan Ahmed",
    "Shahzaib Raza",
    "Muneeb Hassan",
    "Murtaza Ali",
    "Sameer Khan",
    "Adnan Ahmed",
    "Arsalan Raza",
    "Zubair Khan",
    "Kashif Ali",
    "Samiullah Ahmed",
    "Ahsan Khan",
    "Rehan Hussain",
    "Mujtaba Ali",
    "Shayan Ahmed",
    "Azlan Khan",
    "Abrar Hussain",
    "Farhan Ali",
    "Yasir Ahmed",
    "Qasim Khan",
    "Rauf Ali",
    "Sufyan Ahmed",
    "Ammar Khan",
    "Ehtisham Ali",
    "Haris Ahmed",
    "Maham Khan",
    "Areeba Ahmed",
    "Fatima Zahra",
    "Ayesha Khan",
    "Maryam Ali",
    "Hania Ahmed"
]
       #Generate Unique Student IDs
unique_ids = random.sample(range(1000, 9999), 70)
###Generate Unique Student IDs 
students_data = []

for i in range(1, 71):
    student = {
        "Serial_No": i,
        "Student_ID": unique_ids[i - 1],
        "Name": names[i - 1],
        "Marks": random.randint(35, 100)
    }
    students_data.append(student)
##Create and Write Initial CSV
filename = "students.csv"

fieldnames_initial = ["Serial_No", "Student_ID", "Name", "Marks"]

with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames_initial)
    writer.writeheader()
    writer.writerows(students_data)

print("1. CSV file created successfully with 70 student records.")
####Read CSV and Calculate Percentage
updated_students_data = []

with open(filename, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    print("2. CSV file read successfully.")

    for row in reader:
        marks = int(row["Marks"])
        percentage = (marks / 100) * 100
        row["Percentage"] = f"{percentage:.2f}%"
        updated_students_data.append(row)

print("3. Percentages calculated successfully.")
###Save Updated Data Back to CSV
fieldnames_updated = [
    "Serial_No",
    "Student_ID",
    "Name",
    "Marks",
    "Percentage"
]

with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames_updated)
    writer.writeheader()
    writer.writerows(updated_students_data)

print("4. CSV file updated successfully with Percentage column.")
