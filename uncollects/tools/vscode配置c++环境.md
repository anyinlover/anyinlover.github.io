# vscode下的c++环境配置

一直以来都没有搞定vscode的c++环境配置，不知其原理，除了问题也无法定位。其实是没有抓住本质。

vscode的工具链背后其实核心还是命令行的封装。所以应该溯源到命令行上去，才能理解其原理。

核心的几个工具：

cmake：作为编译工具存在，其生成的compile_commands.json需要被clangd和clang-tidy调用。

clangd：作为language server，背后依赖clang的能力，提供代码智能辅助功能。包括报错、补全、定位、大纲、浮窗、格式化、重构等能力。其集成了clangd-tidy和clangd-format的能力。

clangd-tidy：作为linter的角色，帮助识别代码不规范问题。

clangd-fortmat：格式化工具，用于格式化代码。

在vscode中，需要把默认的引擎关闭。vscode插件clangd自动会提醒。其他保持默认配置即可。只要compile_commands.json在常规位置，clangd能够自动找到。如果clangd有报错，仔细看out下的clangd输出，根据报错解决问题。

注意因为clang默认使用的是libstdc++，所以标准库文件一般会关联到gcc的库，除非编译时指定compile args。

clangd-tidy检查格式是允许定制化的，只要根目录下有.clang-tidy。

同样的 clangd-format也可以定制化，调用根目录下的.clang-format。