import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = {
    "StudentID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111],
    "Gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Female"],
    "StudyHours": [5.0, 3.5, 7.0, 2.0, 6.0, 4.0, 8.5, 3.0, 6.5, 4.5, 7.5, 7.5],
    "Attendance(%)": [90, 80, 95, 70, 85, 88, 92, 85, 80, 85, 88, 88],
    "Marks": [78.0, 65.0, 88.0, np.nan, 72.0, 68.0, np.nan, 70.0, 82.0, 84.0, 90.0, 90.0]
}
pd.DataFrame(raw_data).to_csv("student_performance.csv", index=False)

df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
print("\n--- Dataset Info ---")
print(df.info())

missing_marks = df["Marks"].isnull().sum()
duplicate_rows = df.duplicated().sum()
print(f"\nMissing values in Marks: {missing_marks}")
print(f"Duplicate rows: {duplicate_rows}")

df_clean = df.drop_duplicates().copy()
median_marks = df_clean["Marks"].median()
df_clean["Marks"] = df_clean["Marks"].fillna(median_marks)
print(f"After cleaning: {df_clean.shape}")

plt.figure(figsize=(6, 4))
plt.hist(df_clean["Marks"], bins=6, color="seagreen", edgecolor="black")
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.close()

plt.figure(figsize=(5, 4))
plt.boxplot(df_clean["Marks"], patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title("Marks Boxplot")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.scatter(df_clean["StudyHours"], df_clean["Marks"], color="darkorange", edgecolors="black", s=60)
plt.title("StudyHours vs. Marks")
plt.xlabel("StudyHours")
plt.ylabel("Marks")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.close()

corr_matrix = df_clean[["StudyHours", "Attendance(%)", "Marks"]].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

gender_summary = df_clean.groupby("Gender")[["Marks", "Attendance(%)"]].mean()
print("\nAverage Marks and Attendance by Gender:")
print(gender_summary)

print("\nSaved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")