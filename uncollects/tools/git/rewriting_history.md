# Git Rewriting History

## Modifying the Last Commit

To modify the last commit, use `git commit --amend`.

## Modifying Multiple Commits

To modify multiple commit messages, you need to use `git rebase -i HEAD~10`, which allows you to delete, modify, split, or squash commit records.

## filter-branch

`filter-branch` has powerful functionality:

`git fliter-branch --tree-filter 'rm -f passwords.txt' HEAD` can be used to remove a specific file from all historical records in the branch.

`git filter-branch --subdirectory-filter trunk HEAD` can specify a subdirectory as the new root directory.

It can even modify email addresses:

```shell
git filter-branch --commit-filter '
        if [ "$GIT_AUTHOR_EMAIL" = "schacon@localhost" ];
        then
                GIT_AUTHOR_NAME="Scott Chacon";
                GIT_AUTHOR_EMAIL="schacon@example.com";
                git commit-tree "$@";
        else
                git commit-tree "$@";
        fi' HEAD
```
