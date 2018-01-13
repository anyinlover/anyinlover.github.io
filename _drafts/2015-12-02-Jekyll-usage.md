---
layout: post
title: "Jekyll basic usage"
date: 2015-12-2 20:03
categories: productivity
---

## What is Jekyll
> Jekyll is a simple, blog-aware, static site generator. It takes a template directory containing raw text files in various formats, runs it through a converter (like Markdown) and our Liquid renderer, and spits out a complete, ready-to-publish static website suitable for serving with your favorite web server.

## How to install

    gem install jekyll

Download a favorite theme from [jekyllthemes](http://jekyllthemes.org), and unpack all the files into the github pages directory like `~/Documents/anyinlover.github.io`

## Basic Usage

Just write markdown files in _sites, and in the github pages directory input the following command:

    jekyll serve

Then open the 0.0.0.0:4000 in a web browser, you'll find your blogs.

Use git and push it to the github, you'll read your blogs in the Internet like [Anyinlovr's blog](anyinlover.github.io).

## Use mathjax in Kramdown
In `_include/header.html`, add following lines:

~~~
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
                tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
                        });
</script>
<script type="text/javascript"
src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
~~~

Use a pair of $$ in the md file to express latex math.
