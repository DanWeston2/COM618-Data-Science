import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("weekly_attendance.csv")

plt.figure(figsize=(10, 5))

# Group by course_code
for course, group in df.groupby("course_code"):
    plt.plot(group["week"], group["total_attendees"], marker="o", label=course)

plt.title("Weekly Attendance per Course")
plt.xlabel("Week")
plt.ylabel("Attendance")
plt.grid(True)
plt.legend(title="Course")
plt.show()
