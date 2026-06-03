# Data Directory

This directory is intentionally left without datasets, generated files, and model matrices.

Expected local layout:

```text
project/data/detector/datasets/
project/data/detector/derived/
project/data/detector/csv_utils/
project/data/generator/generated_files/
project/data/generator/matrices/
```

The code builds paths relative to `project/`, so the repository can be cloned anywhere as long as this layout is recreated.
