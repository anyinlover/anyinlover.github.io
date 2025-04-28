# Git Basic Snapshotting

## Viewing file status

Use `git status` to view the current state of the repository:

```shell
git status
```

Adding the `-s` option simplifies the output:

```shell
git status -s
```

## Adding files to the staging area

After creating or modifying a file, use `git add` to add it to the staging area:

```shell
git add filename
```

## Viewing file changes

To see differences between modified files and the staging area, use the following command:

```shell
git diff
```

To compare the staging area with the repository, use:

```shell
git diff --staged
```

You can also specify a particular file to compare modifications:

```shell
git diff filename
```

## Adding from the staging area to the repository

Use the `git commit` command to submit staged files to the repository. Use the `-m` flag to provide a commit message:

```shell
git commit -m "add a file"
```

Separating `add` and `commit` into two steps allows for multiple `add` commands before a single `commit`.

Using `git commit -a -m "add all"` automatically stages any modified files and commits them directly.

## Deleting files

To delete tracked files, use `git rm`, followed by `git commit` to finalize the removal:

```shell
git rm filename
```

For removing staged files, the `-f` parameter is required.

If you want to remove a file from the staging area while keeping it in the working directory, use:

```shell
git rm --cached file
```

## Viewing version history

Use `git log` to view the version history. The long string of numbers displayed is the SHA1-computed `commit id`. By default, the version history is sorted in reverse order.

Several parameters are helpful for viewing version history.

`git log -p -2` shows the specific changes made in each version, where the number controls the display count.

`git log --stat` summarizes the changes made in each version.

`git log --oneline` displays a one-line log for each version. When combined with `--graph`, it shows the branching of versions.

`git log --pretty=format` allows for complex control over log formatting.

You can control the time range of logs:

```shell
git log --since=2.weeks
git log --until="2021-9-1"
```

You can also view modifications related to a specific file:

```shell
git log -- path/to/file
```

## Undoing changes

### Modifying a committed version

To make small modifications to the last committed version, such as changing the commit message or adding forgotten files, use:

```shell
git commit --amend
```

### Reverting staged area changes

If changes have already been staged, use the following command to discard those changes:

```shell
git restore --staged filename
```

### Reverting working directory changes

To discard modifications made in the working directory, use the following command:

```shell
git restore filename
```
