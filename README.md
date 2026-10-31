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

An automated fresh-clone check passed on Windows with Python 3.14.0: a new environment, no local data cache, and a deleted result file reproduced the complete saved JSON record. A cached rerun also reproduced the result without network access. These checks were performed by the authoring agent and do not replace the teammate check below.

**Completed (team-reported):** all P.E.K.K.A teammates successfully performed the fresh-clone reproduction check, followed the README instructions, and ran the baseline to create or update `reports/baseline_metrics.json`. This also confirms that all team members can access the repository.

## Run the Lab 04 inference demo

From the repository root, install the pinned requirements as above, then run:

```bash
python src/inference_demo.py --input-ref data/inference_demo_sample/leaf.jpg
```

On Windows, use `.\.venv\Scripts\python.exe src/inference_demo.py --input-ref data/inference_demo_sample/leaf.jpg`. The first run needs internet if the pinned PlantVillage cache is absent. The script reconstructs the same fixed Lab 02 estimator from its training partition because this repository does not contain a serialized model artifact; it does not tune hyperparameters or use held-out evaluation images for fitting. It writes two JSONL lines to `reports/inference_demo.jsonl`: a real attempt on the supplied image and a controlled simulated-unavailable attempt. Each run replaces the prior demo log.

**Safety/validation note:** Lab 03 did not establish a validated uncertainty cutoff. Accordingly, this demo routes every generated category suggestion to human review. It is a classroom inference path, not a validated disease diagnostic; a healthy suggestion does not rule out disease. The input check verifies file readability, RGB mode, format, and minimum dimensions only; it does not reliably detect blur, leaf presence, or unsupported diseases. Review `docs/inference_architecture.drawio` alongside the code and generated records.