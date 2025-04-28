# Git branching

## Creating and Merging Branches

### Basic Branch Creation

First, create and switch to a new branch `iss53`:

`git checkout -b iss53`

The above command is equivalent to the following two steps:

```shell
git branch iss53
git checkout iss53
```

When the new branch `iss53` is created, it points to the same commit as `master`, and `HEAD` now points to `iss53`, indicating the current branch is `iss53`.

![Creating iss53 Branch](https://git-scm.com/book/en/v2/images/basic-branching-2.png)

Now, any modifications and commits made in the working directory are on the `iss53` branch. After one commit, it will look like this:

![Modifying iss53 Branch](https://git-scm.com/book/en/v2/images/basic-branching-3.png)

To switch back to the main branch, use the following command, ensuring the working directory is clean when switching branches:

`git checkout master`

If an urgent fix needs to be handled while on the main branch, you can create a new branch for the fix, and after making the changes, it will look like this:

![Modifying hotfix Branch](https://git-scm.com/book/en/v2/images/basic-branching-4.png)

To merge the content of the `hotfix` branch into the `master` branch, since `master` is the parent of `hotfix`, you can use `Fast-forward` to merge quickly. This means moving the pointer, and `master` will point to `hotfix`'s commit.

`git merge hotfix`

![Merging hotfix](https://git-scm.com/book/en/v2/images/basic-branching-5.png)

At this point, it's okay to delete the `hotfix` branch (it's just deleting a pointer):

`git branch -d hotfix`

### Branch Merging

When we need to merge the `iss53` branch, at this point the `master` branch is no longer the parent of `iss53`, so fast-forward merging cannot be used, and a three-way merge will be used instead.

![iss53 Branch State](https://git-scm.com/book/en/v2/images/basic-merging-1.png)

A three-way merge will generate a new commit:

![Three-Way Merge](https://git-scm.com/book/en/v2/images/basic-merging-2.png)

### Resolving Conflicts

When two branches modify the same file, a merge conflict occurs, and manual resolution followed by recommitting is required.

## Branch Management

The `git branch` command can be used to view all branches, with the current branch indicated by a star.

The command `git branch -v` provides information about the last commit on each branch.

The command `git branch --merged` filters out branches that have been merged into the current branch, which can be safely deleted.

The command `git branch --no-merged` filters out branches that have not been merged into the current branch, and these branches cannot be deleted unless using `-D` for a forced deletion.

`git branch --move bad corrected` renames a local branch.

`git push --set-upstream origin corrected` pushes the new branch to the remote repository.

`git push origin --delete bad` deletes the old branch from the remote repository.

To delete all locally merged branches, you can use the following command:

`git branch --merged | grep -v "master" | xargs git branch -d`

To delete all branches (excluding master), you can use the command:

`git branch | grep -v "master" | xargs git branch -d`

## Branching Workflows

### Long-Running Branches

One workflow involves maintaining a long-running development branch, while the master branch is used only for version releases. This helps maintain version stability through layering.

![Development Branch](https://git-scm.com/book/en/v2/images/lr-branches-2.png)

### Topic Branches

Topic branches allow for feature development, with selective merging at the end:

![Topic Branches](https://git-scm.com/book/en/v2/images/topic-branches-1.png)

## Remote Branches

Git maintains references to remote branches locally. The previous `git remote show origin` command provides information about remote branches, typically named in the format `origin/master`.

![Remote Branches](https://git-scm.com/book/en/v2/images/remote-branches-2.png)

There may be delays in synchronizing remote branches, as they are only fetched when a fetch operation is performed locally. After synchronization, they may appear as follows:

![Fetch Remote Branches](https://git-scm.com/book/en/v2/images/remote-branches-3.png)

When there are multiple remote servers, fetching branches may appear as follows:

![Multiple Remote Branches](https://git-scm.com/book/en/v2/images/remote-branches-5.png)

### Pushing Branches

To push a local branch to a remote branch, use `git push origin serverfix`.

If the remote and local branch names do not match, specify the push command like this: `git push origin serverfix:awesome`.

### Tracking Branches

To create a local branch based on a remote branch and start working, use `git checkout -b serverfix origin/serverfix`. This will create a tracking branch, and when using `git pull`, it will automatically merge correspondingly.

When using the `clone` command, the master branch automatically tracks the remote branch. To set other branches as tracking branches, there are two scenarios:

1. If the local branch does not exist yet, use the above command, or simply use `git checkout serverfix` to automatically create a local tracking branch.
2. If the local branch already exists, use `git branch -u origin/serverfix` on the local branch to set it as a tracking branch.

Use `git branch -vv` to see the correspondence of each branch with its remote branch.

### Pulling Branches

`git pull` essentially performs the combined action of `git fetch` and `git merge`.

## Git Rebase

In addition to the three-way merge mentioned earlier for the `iss53` branch, another approach is rebase merging.

![iss53 Branch State](https://git-scm.com/book/en/v2/images/basic-merging-1.png)

After executing `git rebase master`, the state will be like this:

![Rebased Branch](https://git-scm.com/book/en/v2/images/basic-rebase-3.png)

Essentially, rebase finds the common parent of two branches, saves the new commits of the current branch in a temporary file, then replays the commits from the rebased branch in chronological order, starting from the parent of the branch being rebased. After rebase, the commit history will be clean, with only one merge commit, which is suitable for merging code into the project's main branch. At this point, the master branch only needs a fast-forward merge to incorporate the changes.

The snapshotted result of rebase and merge is the same, but the commit history they generate differs.

### Advanced Rebase Usages

![Multi-branch Development](https://git-scm.com/book/en/v2/images/interesting-rebase-1.png)

In a scenario where you only want to merge the client branch into the main branch, you can use the following rebase command:

`git rebase --onto master server client`

After completion, the effect will be as shown below:

![Multi-branch Rebase](https://git-scm.com/book/en/v2/images/interesting-rebase-2.png)

### Dangers of Rebase

Rebasing a local branch does not have an impact, but if the branch has been published, it can cause significant problems for others. If you have pulled a branch that has been rebased, one remedy is to continue the rebase:

`git pull --rebase`

### Rebase vs. Merge

Rebase and merge are two different methods. Generally, it is recommended to use rebase for releases, while merge suffices for simple projects.
