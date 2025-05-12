
https://docs.python.org/3/tutorial/modules.html

https://docs.python.org/3/tutorial/classes.html#a-word-about-names-and-objects

```mermaid
flowchart TD
    Start[import name] --> CheckCache{name in sys.modules?};
    CheckCache -- Yes --> ModuleFromCache[module = sys.modules[name]];
    ModuleFromCache --> CheckNone{module is None? (already being imported)};
    CheckNone -- Yes --> RaiseCircularImport[Raise ImportError: circular import detected];
    CheckNone -- No --> ReturnModule1[Return module];

    CheckCache -- No --> CallFindSpec[Call importlib.util.find_spec(name)];
    %% find_spec internally iterates sys.meta_path finders (BuiltinImporter, FrozenImporter, PathFinder)
    %% PathFinder itself searches sys.path and uses sys.path_hooks
    CallFindSpec -- find_spec() raises ModuleNotFoundError --> PropagateMNE_from_find_spec[Propagate ModuleNotFoundError];
    CallFindSpec -- find_spec() returns spec (ModuleSpec object) --> SpecAvailable[spec is available];

    SpecAvailable --> CreateModule[module = importlib.util.module_from_spec(spec)];
    %% The module object is created but not yet executed

    CreateModule --> StorePreExec[sys.modules[name] = module];
    %% Module is added to sys.modules *before* execution to handle circular imports

    StorePreExec --> LoaderExists{spec.loader is not None?};

    LoaderExists -- No (e.g., for a namespace package) --> ModuleReady[Module is ready (no execution needed)];
    ModuleReady --> ReturnModule2[Return sys.modules[name]];

    LoaderExists -- Yes --> ExecModule[spec.loader.exec_module(module)];
    %% Loader executes the module's code, populating the module object

    ExecModule -- Success --> ModuleExecuted[Module executed successfully];
    ModuleExecuted --> ReturnModule2;

    ExecModule -- Failure (Exception during execution) --> RemoveFromSysModules[Remove sys.modules[name] if it was added by this import];
    RemoveFromSysModules --> PropagateExceptionFromExec[Propagate Exception];

    %% Subgraph to illustrate the conceptual behavior of find_spec
    subgraph "Conceptual: importlib.util.find_spec(name)"
        direction LR
        FS_Start[Start find_spec for 'name'] --> FS_IterateMetaPath[Iterate finders in sys.meta_path];
        FS_IterateMetaPath --> FS_CallFinderSpec{Current finder's find_spec(name, path, target)?};
        FS_CallFinderSpec -- Returns a ModuleSpec --> FS_SpecFound[Spec found! find_spec() returns this spec];
        FS_CallFinderSpec -- Returns None --> FS_NextFinder[Try next finder];
        FS_NextFinder --> FS_IterateMetaPath;
        FS_IterateMetaPath -- All finders tried, no spec found --> FS_RaiseMNE[find_spec() raises ModuleNotFoundError];
    end

    %% Styling
    style Start fill:#lightgrey,stroke:#333,stroke-width:2px
    style ReturnModule1 fill:#c3e6cb,stroke:#155724,stroke-width:2px,color:#155724
    style ReturnModule2 fill:#c3e6cb,stroke:#155724,stroke-width:2px,color:#155724
    style RaiseCircularImport fill:#f5c6cb,stroke:#721c24,stroke-width:2px,color:#721c24
    style PropagateMNE_from_find_spec fill:#f5c6cb,stroke:#721c24,stroke-width:2px,color:#721c24
    style PropagateExceptionFromExec fill:#f5c6cb,stroke:#721c24,stroke-width:2px,color:#721c24

    classDef process fill:#e2e3e5,stroke:#6c757d,stroke-width:1px;
    classDef decision fill:#ffeeba,stroke:#856404,stroke-width:1px;
    classDef submodule fill:#d1ecf1,stroke:#0c5460,stroke-width:1px;

    class CallFindSpec,CreateModule,StorePreExec,ExecModule,RemoveFromSysModules,ModuleReady,ModuleExecuted,ModuleFromCache process;
    class CheckCache,CheckNone,LoaderExists decision;
    class FS_Start,FS_IterateMetaPath,FS_CallFinderSpec,FS_SpecFound,FS_NextFinder,FS_RaiseMNE submodule;
```
