## Dataset

Please download the test sets for the evaluated problems and benchmark datasets from Google Drive:

```text
https://drive.google.com/drive/folders/1Ptj4a78kZUITdvp7DW3tPw5D0o9wHne5?usp=sharing
```

Place datasets under:

```text
./dataset/
```

For synthetic evaluations, `DataFinder` expects one subdirectory per problem name. For example:

```text
dataset/
├── tsp/
├── cvrp/
├── acvrp/
└── ...
```

For TSPLIB and CVRPLIB, the benchmark tester recursively searches the directory passed through `--data_dir` for `.tsp` or `.vrp` files.
