import subprocess
import sys
import datetime
import os

def run_command(command):
    print(f"\n▶ Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    else:
        print(f"{result.stdout}")

def get_today_branch_name():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"daily-{today}"

def get_commit_message_from_files():
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    if py_files:
        return f"Added {', '.join(py_files)}"
    else:
        return "Daily update"

def checkout_or_create_branch(branch_name):
    # Check if the branch exists
    result = subprocess.run(f"git branch --list {branch_name}", shell=True, capture_output=True, text=True)
    if result.stdout.strip() == "":
        # Branch does not exist, create it
        run_command(f"git checkout -b {branch_name}")
    else:
        # Branch exists, checkout
        run_command(f"git checkout {branch_name}")

def add_commit_push(branch_name, commit_message):
    run_command("git add .")
    run_command(f'git commit -m "{commit_message}"')
    run_command(f"git push origin {branch_name}")

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
    print(f"🌿 Using branch: {branch_name}")

    # Prompt for commit message choice
    use_auto = input("Use auto-generated commit message? (y/n): ").strip().lower()
    if use_auto == 'y':
        commit_message = get_commit_message_from_files()
    else:
        commit_message = input("Enter your commit message: ").strip()
        if not commit_message:
            print("❌ Commit message cannot be empty.")
            sys.exit(1)

    checkout_or_create_branch(branch_name)
    add_commit_push(branch_name, commit_message)

    merge_choice = input("\nDo you want to merge this branch into 'main' now? (y/n): ").strip().lower()
    if merge_choice == 'y':
        merge_branch(branch_name, "main")
        delete_choice = input("Delete the feature branch locally and remotely after merge? (y/n): ").strip().lower()
        if delete_choice == 'y':
            delete_branch(branch_name)

    print("\n✅ All done successfully.")
