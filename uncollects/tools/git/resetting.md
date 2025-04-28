# Git Resetting

`git reset` has three modes.

In soft mode, only the version is rolled back, which moves the branch that HEAD points to back to the parent commit. It is equivalent to undoing the last commit. At this point, the files in the working directory and staging area are still the new version.

![Soft Mode Rollback](https://git-scm.com/book/en/v2/images/reset-soft.png)

Mixed mode replaces the files in the staging area with the old version as well. This is the default mode if no parameters are provided.

![Mixed Mode Rollback](https://git-scm.com/book/en/v2/images/reset-mixed.png)

Hard mode replaces the files in the working directory with the old version as well.

![Hard Mode Rollback](https://git-scm.com/book/en/v2/images/reset-hard.png)
