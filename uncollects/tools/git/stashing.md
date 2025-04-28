# Git Stashing

If you need fix an urgent bug, and the working directory on the existing branch is not clean, you can use the following command to save the current work state:

`git stash`

When you have finished creating the bug branch and merging it into the existing branch, you can restore the work state. The following command can be used to view the saved stash list:

`git stash list`

There are two methods to restore the work state:

`git stash apply`
This does not delete the stash list. You need to use `git stash drop` to delete it.

`git stash pop`
This restores the work state and also deletes the stash list.

In the case of multiple stashes, you can first check the stash list and then restore a specific stash:

`git stash apply stash@{0}`

The stash command has several useful parameters. The first one, `--keep-index`, will keep the changes in the index while stashing. The second, `-u`, will stash untracked files as well.

You can also create a new branch from the stash to avoid potential conflicts: `git stash branch newone`
