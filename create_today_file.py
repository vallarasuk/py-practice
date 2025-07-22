import os
import datetime

def create_today_file(base_path="."):
    now = datetime.datetime.now()
    month_folder = now.strftime("%b")  # Jul, Aug, etc.
    day_file = now.strftime("%d") + ".py"  # 22.py

    # Build paths
    month_folder_path = os.path.join(base_path, month_folder)
    day_file_path = os.path.join(month_folder_path, day_file)

    # Create month folder if it does not exist
    if not os.path.exists(month_folder_path):
        os.makedirs(month_folder_path)
        print(f"📁 Created month folder: {month_folder_path}")
    else:
        print(f"📁 Month folder exists: {month_folder_path}")

    # Check if the date-based .py file exists
    if os.path.exists(day_file_path):
        choice = input(f"⚠️ {day_file} already exists. Do you want to recreate it? (y/n): ").strip().lower()
        if choice != 'y':
            print("✅ Skipped file creation.")
            return
        else:
            print("♻️ Recreating the file...")

    # Create or recreate the file
    with open(day_file_path, "w") as f:
        f.write(f"# {now.strftime('%Y-%m-%d')} Daily Practice\n\n")
        f.write("def main():\n")
        f.write("    pass\n\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    main()\n")
    print(f"✅ File created: {day_file_path}")

if __name__ == "__main__":
    create_today_file()
