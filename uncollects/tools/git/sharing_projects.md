# Git Sharing Projects

## Remote Repository

### Display Remote Repositories

When cloning from a remote repository, Git automatically sets up the local `master` branch to track the remote `master` branch, with the default name for the remote repository being `origin`.

To view information about remote repositories, use `git remote`, and for more detailed information, use `git remote -v`.

### Add Remote Repository

After creating a new repository named "learngit" on GitHub, when a local repository already exists, you can associate the local repository with the remote one using:

`git remote add pb git@github.com:anyinlover/learngit`

### Fetch Content from Remote Repository

Use `git fetch origin` to fetch content from the remote repository to the local repository. Note that this command only fetches remote content and does not perform any merging. When the local and remote branches are linked, `git pull` can be used to fetch and merge automatically.

### Push Content to Remote Repository

To push local repository content to the remote repository: `git push origin master`

### View Remote Repository

`git remote show origin` provides specific information about the remote repository, including the binding and synchronization status of local and remote branches.

### Rename and Delete Remote Repository

To rename a remote repository: `git remote rename pb paul`

To delete a remote repository: `git remote rm paul`

To modify remote url: `git remote set-url upstream ssh://git@xxx`

## Tag Management

When releasing a version, creating a tag uniquely identifies the version at the time of tagging. A tag is a snapshot of the repository and essentially is an unchangeable pointer to a commit.

### View Tags

Use `git tag` to view all tags.

Use `git tag -l "v1.8.5*"` to view tags that match a pattern.

### Create Tags

Create an annotated tag with a message:

`git tag -a v0.1 -m "version 0.1 released"`

To view a tag, use `git tag` to find the tag name and then use the following command:

`git show <tagname>`

Using `-s` allows signing a tag with a private key, utilizing PGP signatures. This requires installing GPG:

`git tag -s v0.2 -m "signed version 0.2 released" commit_id`

Git also supports lightweight tags, which are created by default on the latest commit:

`git tag v1.0`

For tagging historical commits, you need to find the corresponding `commit_id`:

`git tag v0.9 commit_id`

### Push Tags

By default, `git push` does not push tags. To push tags to the remote repository:

`git push origin v1.0`

You can also push all tags to the remote repository:

`git push origin --tags`

### Delete Tags

It is possible to delete incorrectly created tags:

`git tag -d v0.1`

To delete a remote tag:

`git push origin --delete v0.9`

### Switching to a Tag

`git checkout v0.1` can be used to switch to a tag.

Creating a new branch based on a tag:

`git checkout -b version2 v0.2`

## Git Aliases

`git config --global alias.co checkout` allows you to specify git aliases.
