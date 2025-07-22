import subprocess
import sys
import datetime
import os

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    else:
        print(result.stdout)

def get_today_branch_name():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"daily-{today}"

def get_commit_message_from_files():
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    if py_files:
        return f"Added {' ,'.join(py_files)}"
    else:
        return "Daily update"

def create_branch(branch_name):
    run_command(f"git checkout -b {branch_name}")

def add_commit_push(branch_name, commit_message):
    run_command("git add .")
    run_command(f'git commit -m "{commit_message}"')
    run_command(f"git push -u origin {branch_name}")

def merge_branch(branch_name, target_branch="main"):
    run_command(f"git checkout {target_branch}")
    run_command(f"git pull origin {target_branch}")
    run_command(f"git merge {branch_name}")
    run_command(f"git push origin {target_branch}")

def delete_branch(branch_name):
    run_command(f"git branch -d {branch_name}")
    run_command(f"git push origin --delete {branch_name}")

if __name__ == "__main__":
    branch_name = get_today_branch_name()
    print(f"Auto branch name: {branch_name}")

    commit_message = get_commit_message_from_files()
    print(f"Auto commit message: {commit_message}")

    create_branch(branch_name)
    add_commit_push(branch_name, commit_message)
    merge_branch(branch_name, "main")

    delete_choice = input("Delete the feature branch locally and remotely? (y/n): ").strip().lower()
    if delete_choice == 'y':
        delete_branch(branch_name)

    print("✅ All done successfully.")
