# Git Config

After installation, you can set your name, email, and default editor using the following commands:

```shell
git config --global user.name "Michael Gu"
git config --global user.email "anyinlover@gmail.com"
git config --global core.editor vim
```

The `--global` flag is a global parameter, and it's also possible to set different configurations for each Git repository.

You can view all configurations with the following command:

```shell
git config --list
```

To view a specific configuration, you can use:

```shell
git config user.name
```

## Ignoring specific files

By creating a `.gitignore` file in the root directory of a Git working area, you can specify files that you do not want to commit. The [GitHub official website](https://github.com/github/gitignore) provides excellent examples.

Principles for ignoring files:

- Ignore auto-generated files, such as thumbnails.
- Ignore compiled intermediate files, executables, etc.
- Ignore personal configuration files containing sensitive information.
