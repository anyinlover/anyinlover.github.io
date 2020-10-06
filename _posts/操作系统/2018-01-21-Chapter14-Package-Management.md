---
title: Package Manager
category: 操作系统
tags:
  - Linux
---

The most important determinant of distribution quality is the packaging system and the vitality of the distribution's support community.

## Packaging Systems

Major Packaging System Families

|  Packaging System   | Distributions (Partial Listing)                                         |
| :-----------------: | :---------------------------------------------------------------------- |
| Debian Style(.deb)  | Debian, Ubuntu, Xandros, Linspire                                       |
| Red Hat Style(.rpm) | Fedora, CentOS, Red Hat Enterprise Linux, OpenSUSE, Mandriva, PCLinuxOS |

## How A Package System Works

### Package Files

package maintainer

upstream provider

### Repositories

### Dependencies

### High And Low-level Package Tools

Low-level Tools: handle tasks such as installing and removing package files

High-level Tools: perform metadata searching and dependency resolution

Packaging System Tools

|              Distributions               | Low-Level Tools | High-Level Tools  |
| :--------------------------------------: | :-------------: | :---------------: |
|               Debian-Style               |      dpkg       | apt-get, aptitude |
| Fedora, Red Hat Enterprise Linux, CentOS |       rpm       |        yum        |

## Common Package Management Tasks

Low-level tools support creation of package files

### Finding A Package In A Repository

Package Search Commands

|  Style  |                   Command(s)                   |
| :-----: | :--------------------------------------------: |
| Debian  | apt-get update; apt-cache search search_string |
| Red Hat |            yum search search_string            |

### Installing A Package From A Repository

|  Style  |                  Command(s)                  |
| :-----: | :------------------------------------------: |
| Debian  | apt-get update; apt-get install package_name |
| Red Hat |           yum install package_name           |

### Installing A Package From A Package File

Notice: No dependency resolution is performed

Low-Level Package Installation Commands

|  Style  |         Command(s)          |
| :-----: | :-------------------------: |
| Debian  | dkpg --install package_file |
| Red Hat |     rpm -i package_file     |

### Removing A Package

Packages can be uninstalled using either the high-level or low-level tools

Package Removal Commands

|  Style  |         Command(s)          |
| :-----: | :-------------------------: |
| Debian  | apt-get remove package_name |
| Red Hat |   yum erase package_name    |

### Updating Packages From A Repository

|  Style  |           Command(s)            |
| :-----: | :-----------------------------: |
| Debian  | apt-get update; apt-get upgrade |
| Red Hat |           yum update            |

### Upgrading A Packages From A Package File

Low-Level Package Upgrade Commands

|  Style  |         Command(s)          |
| :-----: | :-------------------------: |
| Debian  | dpkg --install package_file |
| Red Hat |     rpm -U package_file     |

### Listing Installed Packages

Package Listing Commands

|  Style  | Command(s)  |
| :-----: | :---------: |
| Debian  | dpkg --list |
| Red Hat |   rpm -qa   |

### Determining If A Package Is Installed

Package Status Commands

|  Style  |         Command(s)         |
| :-----: | :------------------------: |
| Debian  | dpkg --status package_name |
| Red Hat |    rpm -q package_name     |

### Displaying Info About Installed Package

Package Information Commands

|  Style  |         Command(s)          |
| :-----: | :-------------------------: |
| Debian  | apt-cache show package_name |
| Red Hat |    yum info package_name    |

### Finding Which Package Installed A File

Package File Identification Commands

|  Style  |       Command(s)        |
| :-----: | :---------------------: |
| Debian  | dpkg --search file_name |
| Red Hat |    rpm -qf file_name    |
