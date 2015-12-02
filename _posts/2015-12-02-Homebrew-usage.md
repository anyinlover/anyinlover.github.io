---
layout: post
title: "Homebrew basic usage"
date: 2015-12-2 7:29
categories: [productivity, system]
---

> The missing package manager for OS X

## What is Homebrew

* Homebrew installs the stuff you need that Apple didn’t.
* Homebrew installs packages to their own directory and then symlinks their files into /usr/local.

## How to install Homebrew
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Just following the script explains.

## Basic Usage

### Install

    brew install wget

### Uninstall 

    brew uninstall wget

### List what have been installed

    brew list

### Search which can be installed

    brew search wget

### Update Homebrew

    brew update

### Update formulae

    brew upgrade

### Diagnose Homebrew

    brew doctor
    
### Homebrew help

    brew help

### Homebrew manual

    man brew
