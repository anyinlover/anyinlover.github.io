---
tags:
  - ruby
  - bundler
  - gem
---

# ruby & gem & bundler

## ruby

Ruby was created by Matz, it is designed as a balance language just as its creator said:

> I wanted a scripting language that was more powerfull than Perl, and more object-oriented than Python.

So in Ruby everything is an Object, similar to Java.

Ruby has some other features but I think they my be advanced 20 years ago, now are common.

Ruby become popular for the web framework written in it: Ruby on Rails.

### Ruby on Rails

Rails is a web application development framework. In Java, there are Apache Struts and Spring. In Python, there is Django.

GitHub, Airbnb, Hulu all are built with Ruby on Rails.

The Rails philosophy includes two major guiding principles:

* Don't Repeat Yourself
* Convention over configuration

## rbenv

rbenv is a great way to manage several ruby versions. In fact, Python also have a pyenv, the usage is totally same, even the guide too. When I want to install ruby, We don't need the offical RVM. I use the following way:

``` shell
brew install rbenv #auto install ruby-build
rbenv install -l # lookup the latest version
rbenv install 2.5.0
rbenv global 2.5.0
ruby -v
```

And don't forget to set up rbenv integration with the shell. Just append the line:

`eval "$(rbenv init -)"`

We can set ruby version in local directory by using:

`rbenv local 2.5.0`

To unset local version:

`rbenv local --unset`

To list all versions:

`rbenv versions`

To display current active version:

`rbenv version`

To uninstall ruby version:

`rbenv uninstall 2.5.0`

## RubyGems

RubyGems is a package management framework for Ruby. Here we use it to install Bundler, Jekyll and all other things.

`gem install bundler`

gem have some other options, like list, search, update, uninstall.

The offical RubyGems mirror is very slow in China, so we can change mirror:

``` shell
gem sources -l
gem sources --add https://gems.ruby-china.org/ --remove https://rubygems.org/
```

If you want to update gem itself, using:

`gem update --system`

## Bundler

Bundler provides a consistent environment for Ruby projects by tracking and install the exact gems and versions that are needed.

In a word, Bundler deal with dependency.

Bundle use a Gemfile to control the dependencies in the project directory.

First generate a Gemfile:

`bundle init`

Then write the dependencies into the Gemfile, and install it:

`bundle install # bundle also fine`

If you want to update the gems to the last version, use update (carefully);

`bundle update`

Bundler have its own mirror setting, by setting:

`bundle config mirror.https://rubygems.org https://gems.ruby-china.org`

The source in Gemfile will be replaced.
