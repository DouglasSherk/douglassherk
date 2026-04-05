---
layout: post
title: "Machine learning engineering projects are DAGs"
date: 2020-10-13
categories:
  - Data Engineering
  - Data Science
  - Machine Learning
  - Software Engineering
source_url: https://datastronomy.com/machine-learning-engineering-projects-are-dags/
---

The team gets started on a machine learning engineering project, and they write scripts that they then throw into a shared repository with little thought as to how everything will fit together. At first, things seem fast and smooth. The team flies through the problem. They're agile.

But they inevitably find as the project progresses that they need to revisit initial assumptions and iterate. Mistakes happen, but to correct them, the team needs flexibility. Let's say they want to rebuild their model pipeline's extracted features to include additional features they initially missed, so... they try.

## A sad state of affairs

Jake is working on the feature extractor and needs to use the scraper that Mary built to recreate the dataset. He fires up that component, but he can't make heads or tails of how it works. The code has hardcoded paths like `/home/mary/repos/our_team_project/tmp/datasets`. The scraper depends on a different version of SciKit-Learn than the one the feature extractor uses. The code requires access to a third-party service that Mary forgot to check into the repository.

Most egregious of all, it turns out that scraper depends on the feature extractor's outputs! Jake facepalms as he realizes that he checked in the extracted features directly into the repository and Mary, rightly so, assumed that it was safe to build the scraper on top of what was already there. The unfortunate state of affairs is as follows:

![A circular dependency between a scraper and feature extractor.](/assets/posts/machine-learning-engineering-projects-are-dags/circular-dependency.svg)

*There's now a circular dependency that becomes difficult to untangle.*

There's now a **circular dependency** that becomes difficult to untangle. Nobody can build the project, and everyone is sad. The project grinds to a halt and the team needs days to get back on-track. If only Jake and Mary had spent an hour or so coordinating things ahead of time, or someone had laid out a framework for them that avoided such situations, this would have never happened!

## A flawed mindset

What leads to situations like these, aside from laziness, is that we like to believe as machine learning engineers that we're tackling novel problems that nobody has before. We desperately cling to the idea that we're mad scientists in search of truth. We wish to believe that through sheer brain power, we'll discover the answer, and someone else can sort out how to deploy it. But rarely are any of those the case.

Most often, we're building simple SVMs and regressions. Maybe we're even dabbling with deep learning, but even then, we're usually solving problems in well-understood spaces and with known patterns such as object detection or sentiment analysis.

Here's the secret I'm giving you in this article: machine learning engineering projects don't need to be much different from traditional software engineering projects. Sure, there's more experimentation than there is in traditional software engineering, but I would argue that:

> Most machine learning engineering projects suffer from sociological and logistics risks rather than technological risks.

That is, the primary risk is not that you'll fail to extract useful signal and build a predictive model, but that you won't be able to work together as a team, or collect the data in the required format, or iterate until you arrive at useful results.

## A better mental model

Remember when you used to work on traditional software engineering projects? You had the following tools that made your life easier:

- A button you could press to build the entire project so you instantly knew where there were compilation and/or linker errors.
- A cleaning process where you could destroy intermediates and be sure that you can build the project from scratch.
- Continuous integration with other developers, so you knew everyone was building the same thing.
- `Makefile`s or other artefacts that you could update to alter the build process on others' behalf without them needing to understand those changes.

*You can still have all of these things* when you work on machine learning engineering projects. Let's see how.

## The antidote: DAGs

The antidote to the flawed mindset is to think of machine learning engineering projects as **directed acyclic graphs** (DAGs). A DAG is a graph data structure where:

- All nodes are connected by directed edges, so they go one way, and not the other.
- There are no cycles in the graph. From each node, there's no path you can take to get back to that node.

The key is to model your project's artefacts as nodes and dependencies as directed edges. As long as this graph is a DAG, then you're doing it right.

Of course, this mental model alone won't solve all of these problems, but it's the foundation upon which all other improvements lie.

![A machine learning workflow expressed as a DAG.](/assets/posts/machine-learning-engineering-projects-are-dags/project-dag.svg)

*An example DAG modeling a typical machine learning engineering workflow.*

Let's get more practical and see what else we need to do.

### One-touch project build

One-touch build in traditional software engineering projects requires that dependencies be a DAG data structure. Your module A can't circularly depend on module B, or you'll get a compilation error.

The difference is that instead of code modules, headers, and links depending on each other, our intermediate data representations must be a DAG.

You should be able to build the entire project from scratch with a single command. You can use `Makefile`s or anything else. The gold standard is when you can run something as simple as this, and the entire project will build:

```make
make download_from_s3 # downloads initial data from S3
                      # if there is any
make                  # builds everything!
```

Using `Makefile`s, this is easy to accomplish by declaring your dependencies:

```make
raw:
    python -m raw data/raw/

preprocess: data/raw/
    python -m preprocess data/raw/ data/preprocessed/

features: data/preprocessed/
    python -m features data/preprocessed/ data/features/

model: data/features/
    python -m model data/features/ models/model.pkl

all: raw preprocess features model
```

We can also diagram this `Makefile` as follows:

![A clean step-by-step workflow with clear inputs and outputs.](/assets/posts/machine-learning-engineering-projects-are-dags/makefile-workflow.svg)

*A clean, step-by-step workflow with clear inputs and outputs.*

### Data is immutable

Once you write a dataset, you should never alter it. When you change data after it's written, it becomes difficult to backtrack, understand the control flow, and ensure that your team is all working with the same artefacts. If you need to transform your data, it's better to add an additional step to the workflow.

Another advantage of making data immutable is that it takes better advantage of intermediates. For example, suppose you discover that you made a mistake in the 3rd step of your pipeline, and need to iterate on it. Rather than having to rebuild the dataset from scratch, you can just discard intermediates from the 3rd step onward. This workflow enables you to iterate faster and more confidently.

### Each step has clear inputs and outputs

Notice also how in the above `Makefile`, each step of the process specifies dependencies, consumes an input and produces an output. For example, the `preprocess` step depends on `data/raw/` which the `raw` step produces, and it outputs the `data/preprocessed/` data for the next step, `features`.

Being clear about inputs and outputs means that each component and dataset is responsible for one thing, which leads to cleaner and more modular code. It also makes it easier to swap out components and datasets when necessary.

### Build from scratch or from intermediates

Your pipeline should enable you to iterate fast from any step in the workflow. You should be able to both:

1. Build the project completely from scratch with no errors, missing dependencies, or differences with others on your team.
2. Build from any step of the process onward assuming you already have the intermediates.

A good way to accomplish both of these goals is to have a `make clean` command or equivalent that destroys all intermediates and to run it often and/or in CI to make sure that the whole project works end-to-end. If you write your `Makefile` correctly, it will automatically detect when intermediates are present and skip the unnecessary steps when you run your `make` command.

### Dependencies are part of the DAG

Just because we're focused on data dependencies doesn't mean we should ignore our software dependencies! Software dependency management is just as important as data dependency management.

In general, there are two ways to wrangle your dependencies:

1. Use one common environment for all steps, e.g., one shared Anaconda environment.
2. Put each step into its own isolated environment, e.g., Docker containers.

Either solution is fine and depends on the scale of your problem. For a smaller project that you won't maintain as long, a single shared environment could serve you well, while Docker containers would be overkill. But if you're working across many verticals with conflicting dependencies, then Docker containers suddenly become more attractive.

## Conclusion

Just like when building traditional software, when you think through your machine learning project workflow before you get started, you'll save tremendous future headaches.

If you're interested in learning more, check out the [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) framework, which explains this thinking and much more.
