# Git Revision Selection

## Short SHA-1

Generally, using the first 8-10 characters of a SHA-1 can avoid ambiguity.

## Branch Reference

If a branch points to a specific commit, you can directly use branch operations. `git rev-parse topic1` can be used to view the commit corresponding to the branch.

## Reflog

The reflog (`git reflog`) retains the history of HEAD and branch references for the past few months, making it convenient for operations like resetting.

We can use `git show HEAD@{5}` or `git show master@{yesterday}` to refer to specific commits.

The reflog is only kept in the local repository.

## Ancestor References

`HEAD^n` represents the nth parent node, by default n is 1. This is especially useful in merge nodes.

`HEAD~n` represents the nth ancestor commit, by default n is 1. `HEAD~~` represents `HEAD~2`.

## Commit Ranges

`git log master..experiment` shows commits that are in the experiment branch but not in the master branch.

Both `git log ^master experiment` and `git log experiment --not master` have the same meaning.

`git log master...experiment` displays all non-shared commits between the two branches.
