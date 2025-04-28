# Distributed Git

## Distributed Workflows

Here are three common distributed workflows:

![Centralized Workflow](https://git-scm.com/book/en/v2/images/centralized_workflow.png)
![Integration-Manager Workflow](https://git-scm.com/book/en/v2/images/integration-manager.png)
![Dictator and Lieutenants Workflow](https://git-scm.com/book/en/v2/images/benevolent-dictator.png)

## Committing to a Project

### Commit Guidelines

Use `git diff --check` to check for whitespace issues.

Strive to make small, focused commits.

Consider rewriting commit history.

Follow the repository's commit message format.

### Small Private Team

Before pushing, merge the latest code from the main repository. This workflow is simple.

![Small Private Team](https://git-scm.com/book/en/v2/images/small-team-6.png)

### Managed Private Team

Only committers can merge code. For contributors, typical commit records look like this:

![Managed Private Team](https://git-scm.com/book/en/v2/images/managed-team-3.png)

To see changes made by others during this period, use the following command:

`git log --no-merges issue54..origin/master`

### Forking Public Projects

You need to fork separately, and the main repository will remain clean.

![Forking Public Projects](https://git-scm.com/book/en/v2/images/public-small-3.png)

```shell
git clone git@github.com:guguoshenqi/gitskill.git
git checkout -b featureA
git remote add myfork git@github.com:myfork/gitskill.git
git push -u myfor featureA
git request-pull origin/master myfork
```

In general, to maintain a clean main repository commit history, you can use rebase:

```shell
git checkout featureA
git rebase origin/master
git push -f myfork featureA
```

If code review finds areas that need modification, the following method is often used to combine all changes into one commit record:

```shell
git checkout -b featureBv2 origin/master
git merge --squash featureB
... change something ...
git commit
git push myfork featureBv2
```

## Maintaining a Project

### Using Feature Branches

Newly merged code is recommended to start in a feature branch, with namespaces for multi-person cases:

`git branch ghs/fun master`

### Viewing Changes

`git log featureA --not master` can be used to view new commit records.

`git diff master...featureA` can be used to view specific file changes.

### Integration Processes

There are various integration processes to choose from.

The merging process is the simplest, maintaining only the master daily branch, which can lead to bugs.

![Merging Workflow](https://git-scm.com/book/en/v2/images/merging-workflows-2.png)

The two-phase merge process solves bug issues by introducing a development branch.

![Two-Phase Merge Workflow](https://git-scm.com/book/en/v2/images/merging-workflows-4.png)

Some projects use more complex integration processes, such as the Git project.

Another way of integrating code is through rebasing or cherry-picking. Rebasing has been introduced above, while cherry-picking allows selecting a specific commit to rerun:

![Picking a Topic Branch](https://git-scm.com/book/en/v2/images/rebasing-2.png)

`git cherry-pick e43a6`

By default, Git uses the `Fast forward` mode, which discards branch information after deleting it. To disable the Fast forward mode merge and create a new commit to retain the deleted branch information, use the `--no-ff` parameter:

`git merge --no-ff -m "merge with no-ff" dev`
