<!-- markdownlint-disable-next-line -->
# <img src="docs/logo.png" alt="logo" width="40" style="vertical-align:middle;"/> Beyond the Blanket Challenge

![Final leaderboard summary](/labs/05_final_eval/fig/team_metric_summary.png)

<p align="center">
  <a href="https://huggingface.co/datasets/CSE472-blanket-challenge">🤗 Datasets</a> |
  <a href="docs/READING.md">📚 Reading</a> |
  <a href="labs/05_final_eval/README.md">🏆 Final Results</a> |
  <a href="CHANGELOG.md">📝 Changelog</a>
</p>

## About

This repository contains the materials and code for the CSE472 (Fall 2025) course
project "Beyond the Blanket Challenge".
The project is designed by [Shu Wan](https://shu-wan.github.io/).

The materials here include datasets, evaluation scripts, and a summary of the
final leaderboard and results.

> [!NOTE]
>
> This project is finalized.

---

## :trophy: Leaderboard

|        Rank         | Group | Members           | Final Score (↓) |
| :-----------------: | :---: | ----------------- | :-------------: |
| :1st_place_medal: 1 |  10   | Ang, Muhammed     |  **0.194036**   |
| :2nd_place_medal: 2 |   8   | Fredo, Anton      |    0.208636     |
| :3rd_place_medal: 3 |   7   | Dhruv, Sahajpreet |    0.280585     |
|  :medal_sports: 4   |   1   | Sameera, Tanmayi  |    0.304475     |

## Overview

**Feature selection** is a critical step in machine learning, directly impacting model
performance, interpretability, and generalization capabilities.

**Causal Feature Selection (CFS)** tackles the challenge by selecting features based on their causal
relationships with the target variable. It has been shown that a set of features known as the
**Markov blanket**, provide optimal predictability to the target.
However, recent work shows that many CFS methods are outperformed by non-causal models,
and struggle to generalize across domains.

**Your Mission**: Design a model that performs causal feature selection and achieves
robust performance across different environments.

### Timeline & Milestones

Project Duration: 7 Weeks (Oct 10–Nov 27, 2025)

| Phase                | Duration           | Focus Areas              |
| :------------------- | :----------------- | :----------------------- |
| **Warm-up**          | Oct 10–21, 2025    | Reading / Implementation |
| **Phase 1**          | Oct 22–30, 2025    | Implementation / Tasks   |
| **Phase 2**          | Oct 31–Nov 5, 2025 | Implementation / Tasks   |
| **Phase 3**          | Nov 6–12, 2025     | Implementation / Tasks   |
| **Final Evaluation** | Nov 13–27, 2025    | Implementation / Report  |

See detailed schedule in [SCHEDULE.md](docs/SCHEDULE.md)

## Environment Setup

I already added required environment settings in `pyproject.toml`. You can use uv to set up the environment:

```bash
uv venv  # start a virtual environment
source .venv/bin/activate  # activate the virtual environment
uv sync --all-groups # sync with pyproject.toml
```

conda, pip, or other environment management tools can also be used.

## 📚 Resources & Materials

* [🤗 Hugging Face Datasets](https://huggingface.co/datasets/CSE472-blanket-challenge): our project
Hugging Face space, hosting datasets, models and more
* [📚 Reading Materials](docs/READING.md): Reading materials and where to download them
* [📝 Changelog](CHANGELOG.md)

## 🎯 Deliverables

| **Deliverable** | **Description** |
|:---|:---|
| **Challenge Submission** | Complete code, model, and reproduction instructions |
| **Project Report** | Comprehensive analysis (template to be provided) |
| **Additional Materials** | Any course-specific requirements |

## Frequently Asked Questions

<details>
<summary><strong> If the project goes well, could students be included in a paper,
or is it mainly about replication?</strong></summary>

The project builds on an ongoing research that's still in an exploratory stage.
The instructor will ensure all tasks have reasonable, verifiable answers and will be actively
involved in the process. While the main goal is to explore and understand the work, there is
potential for meaningful contributions depending on progress.
</details>

<details>
<summary><strong> I'm not enrolled in the course or couldn't join a team. Can I still participate?</strong></summary>

**Yes!** All materials are public—feel free to follow along and ask questions.
We welcome independent learners and contributors.

</details>

<details>
<summary><strong> Can I use LLMs for this project?</strong></summary>

**Absolutely!** LLMs are encouraged for brainstorming and assistance. However, you must:

* Disclose where they were used in your final report
* Verify the correctness of any generated content
* Take full responsibility for all submitted work

</details>

<details>
<summary><strong> Can teams discuss the project with each other?</strong></summary>

Yes, but only discuss high-level ideas and concepts. Do not share code or specific implementation details.
All submissions must be your own work

</details>

<details>
<summary><strong> How much compute do I need?</strong></summary>

**Moderate requirements:**

* Access to Sol should be sufficient
* Google offers free Colab Pro for student accounts
* Most experiments can run on standard hardware

</details>

<details>
<summary><strong> What do winners receive?</strong></summary>

**Glory and more:**

* :trophy: Glory, :point_up: Eternal bragging rights
* Potential research and publication opportunities

</details>

## Disclaimer

This project is an independent educational project created for learning and research purposes.
It is not an official ASU course resource, and it is not affiliated with or
endorsed by Arizona State University.
All content, datasets, and code are provided as-is for public educational use.

---

<p align="center">
  <a href="https://github.com/DMML-ASU/DMML/wiki">
     <img src="https://avatars.githubusercontent.com/u/30509390?s=200&v=4"
         alt="dmml_logo"
         height="30"
         style="vertical-align:middle; margin-right:10px;" />
  </a>
  <sub>2025 Built with ❤️.</sub>
</p>
