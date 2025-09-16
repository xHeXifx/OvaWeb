import requests
import git
import os
import shutil
from art import *


# declare constants
repo_addr = 'https://github.com/xHeXifx/OvaWeb.git'
gist_id = "8db2e2960486e575efce866f89e9334c"
current_dir = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(current_dir, "tmp")

def pull_repo(repo, dir):
    if os.path.isdir(tmp_dir):
        print(f"Removing folder '{tmp_dir}' as it already exists.")
        shutil.rmtree(tmp_dir)
    try:
        print(f'Cloning repo: {repo}')
        # use gitpython to clone the repo easily
        repo = git.Repo.clone_from(repo, dir)
    except Exception as e:
        print(f"Error pulling repo, attempt a fix and re-run script.\n{e}")

def copynew():
    # variables for new file paths for shutil
    newapppy = os.path.join(tmp_dir, "app.py")
    newindex = os.path.join(tmp_dir, "templates", "index.html")
    newVERSION = os.path.join(tmp_dir, 'VERSION')

    print("Copying new app.py, index.html and VERSION from tmp")

    try:
        shutil.copy2(newapppy, current_dir)
        shutil.copy2(newindex, os.path.join(current_dir, "templates"))
        shutil.copy2(newVERSION, current_dir)
        print("Successfully copied over new app.py, index and VERSION")
    except Exception as e:
        print(f"Error copying new files from temp.\n{e}")

def removetmp():
    print("Cleaning up ...")
    shutil.rmtree(tmp_dir)

def getVer():
    # make gist api url using gistid and return it
    req = f'https://api.github.com/gists/{gist_id}'
    res = requests.get(req)
    if res.status_code == 200:
        gdata = res.json()
        for filename, file_info in gdata["files"].items():
            version = file_info["content"]
            return version
        else:
            print(f'Failed to fetch version data from gist: {res.status_code}')


if __name__ == '__main__':
    tprint("OvaWeb Updater")
    print("")

    latestVersion = float(getVer())
    
    if os.path.exists(os.path.join(current_dir, "VERSION")):
        with open(os.path.join(current_dir, "VERSION"), "r") as f:
            currentVersion = float(f.read())
    
    if latestVersion > currentVersion:
        print("OvaWeb can be updated!")
        print(f"Current Version: {currentVersion}")
        print(f"\033[92mLatest Version: {latestVersion}\033[0m")

        print('')

        print('Would you like to update? (Y/N)')

        usrInput = input("> ").lower()

        if usrInput == 'y':
            print('')
            print('\nStarting Update...')

            pull_repo(repo_addr, tmp_dir)
            copynew()
            removetmp()

            print('')
            print('Update complete. New files have been copied to root.')
        elif usrInput == 'n':
            print("Cancelling update.")
        else:
            print('Invalid input.')