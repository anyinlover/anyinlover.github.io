# Import System

```mermaid
flowchart TD
    Start((Start))
    Import[import p.name]
    CheckCache{p.name in sys.modules?}
    GetModule{"module = sys.modules[p.name]"}
```
