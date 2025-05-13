# Use Python Import System to Patch

In many occasions, we need to patch a third-party library in python. A common way to do it is using monkey patch. However, monkey patch is not a perfect way, for it changes attrs dynamically after the module is imported. In sometimes, the modified module has been imported before it is changed, so the monkey patch is not work.

A better way is using python import system to achieve it. Python import a module by three steps.

1. Search the module by Finder
2. Create the module by Loader
3. Bind the module in the current namespace
