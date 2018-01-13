---
layout: post
title: "rbenv Basic Usage"
date: 2015-12-2 8:27
categories: [productivity, language]
---

## What is rbenv

rbenv is used to install and pick a ruby version for your applications.

## How to install

    brew install rbenv ruby-build

In fact, rbenv is used to pick a ruby version, and ruby-build for install.

## Basic usage

### Install Ruby Versions

    # list all available versions:
    rbenv install -l

    # install a Ruby version:
    rbenv install 2.2.3

### Uninstall Ruby Versions

    rbenv uninstall 2.2.3

### Set a local Ruby version

    rbenv local 2.2.3

### Show the local Ruby version

    rbenv local

### Set a global Ruby version

    rbenv global 2.2.3

### Show the global Ruby version

    rbenv global

### List all Ruby versions

    rbenv versions

### List the active version

    rbenv version

### Rehash after a new installation

    rbenv rehash

### rbenv help
    
    rbenv help
