# Tomato Disease Detection

- **Team name:** P.E.K.K.A
- **Team members:** Sobolev Pavel, Gussev Arsenii, Nadobnov Mikhail, Kiseleva Veronika
- **Communication:** Telegram chat group

We want to help small-scale tomato growers use photos of tomato leaves to flag possible disease symptoms and decide which plants need closer inspection or expert advice.

Our planned starting data is the tomato-only subset of [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), including healthy tomato leaves and the available tomato disease categories.

See the [project brief](docs/project_brief.md) for the proposed workflow, initial data plan, open questions, and three next actions.

Lab 01 established the direction; Lab 02 adds a small, reproducible baseline. This is a course experiment. The proposed grower-facing system has not been built or deployed.

## Run the Lab 02 baseline

Use **Python 3.14** (tested with Python 3.14.0 on Windows). From the repository root, create an isolated environment and install the dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run this exact command from the repository root:

```powershell
.\.venv\Scripts\python.exe src/baseline_pipeline.py
```

The command downloads and checks a fixed sample of 320 tomato images across eight categories, trains logistic regression on RGB color histograms, evaluates 80 held-out images, and automatically creates or replaces `reports/baseline_metrics.json`. The first run requires internet; subsequent runs reuse the ignored local data cache. See [data instructions](data/README.md) for sources, placement, excluded categories, and permissions, and [split strategy](docs/split_strategy.md) for the evaluation decision. No full dataset, trained model file, or GPU is needed.

On macOS/Linux with Python 3.14 installed, use `python3.14 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, and `.venv/bin/python src/baseline_pipeline.py`.

The recorded baseline achieved **62.5% accuracy (50/80 held-out images)**; see [saved metrics](reports/baseline_metrics.json). Its main limitation is that a small sample of controlled PlantVillage images does not establish performance on growers' phone photos; color features may exploit backgrounds or lighting.

## Teammate reproduction check

**Pending:** a teammate who did not write the pipeline must make a fresh clone in a new folder, follow the setup and run commands above without verbal instructions, and confirm that `reports/baseline_metrics.json` is created or updated. Share the result or setup problem in the team's Telegram chat. Every member's repository access also still needs confirmation.
