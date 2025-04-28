# XDG Base Directory Specification

[XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/) defines where user-relative files should be looked for by defining one or more base directories relative to which files should be located.

| Env                | Specification                                                              | Default                         |
| ------------------ | -------------------------------------------------------------------------- | ------------------------------- |
| `$XDG_DATA_HOME`   | user-specific data dir                                                     | `$HOME/.local/share`            |
| `$XDG_CONFIG_HOME` | user-specific configuration dir                                            | `$HOME/.config`                 |
| `$XDG_STATE_HOME`  | user-specific state dir                                                    | `$HOME/.local/state`            |
| `$XDG_CACHE_HOME`  | user-specific non-essential data dir                                       | `$HOME/.cache`                  |
| in `PATH`          | user-specific executable dir                                               | `$HOME/.local/bin`              |
| `$XDG_RUNTIME_DIR` | user-specific runtime dir                                                  |                                 |
| `$XDG_DATA_DIRS`   | the preference-ordered set of base directories to search for data          | `/usr/local/share/:/usr/share/` |
| `$XDG_CONFIG_DIRS` | the preference-ordered set of base directories to search for configuration | `/etc/xdg`                      |

The `$XDG_STATE_HOME` contains state data that should persist between (application) restarts, but that is not important or portable enough to the user that it should be stored in `$XDG_DATA_HOME`. It may contain:

- actions history (logs, history, recently used files, …)
- current state of the application that can be reused on a restart (view, layout, open files, undo history, …)
