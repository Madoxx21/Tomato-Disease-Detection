# Project brief

## User and situation

Our proposed users are small-scale tomato growers who notice unusual spots, discoloration, or other visible symptoms on tomato leaves. They may compare photographs with reference material or ask someone more experienced, but similar-looking symptoms can make the next step unclear. We need to confirm this workflow with a grower.

We want to help growers identify leaves that deserve closer inspection and prepare a possible disease category to discuss with an expert. The semester project will focus on a small prototype for individual tomato-leaf photos and the categories supported by our selected data. It will not establish a confirmed diagnosis or prescribe treatment.

## What the system should do

- **Input:** One clear color photograph of a tomato leaf, submitted by the user.
- **Useful output:** A suggested category among the supported tomato disease and healthy-leaf labels, or an indication that the image needs review or a clearer photo. The interface should explain that the suggestion is preliminary.
- **Action or decision after the output:** The grower decides whether to inspect the plant more closely, take another photograph, or seek expert advice before choosing any treatment.

## Why AI may help

The system may need to learn combinations of leaf color, spots, texture, and shape from labeled examples. These patterns can vary between photographs and may be difficult to describe using fixed rules alone.

The future system also needs a photo-upload interface, guidance for taking a usable photo, basic file checks, readable category descriptions, and a way to request human review when a result is unclear. Growers and experts remain part of the decision process. We have not chosen a model or established that dataset performance would transfer to growers' photographs.

## Initial data plan

- **Where the data may come from:** [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage). We plan to extract only tomato examples, retaining healthy leaves and available tomato disease labels. We intend to start with the original color images and exclude other crops and alternative image versions.
- **What we can access now:** The public dataset card and file-path preview. The card describes image and label fields, crop information, and a color configuration, and lists the license as CC BY-SA 3.0. We have not downloaded or extracted images, checked class counts, or verified a usable tomato sample. No dataset will be added to the Lab 01 repository.
- **What still needs confirmation:**
  - Can we obtain actual color tomato images with reliable labels from this source, and what attribution and reuse conditions must we follow?
  - Is the available tomato subset manageable for the course, and can we later obtain a small, permitted sample of realistic grower photographs to check whether the proposed input is suitable?
  - Does a grower find the proposed output useful for deciding what to inspect or ask an expert about?

## Three next actions

1. Inspect the source's file structure and reuse terms, then confirm access to a small sample of actual tomato images and matching labels outside the repository. Check the healthy and disease categories and whether color images can be selected without duplicate image versions.
2. Discuss the proposed photo-to-review workflow with one tomato grower or gardening adviser. Confirm the useful next decision and whether a small sample of typical leaf photographs could be shared with permission; narrow the scope if necessary.
3. Agree on the team name and communication space, fill in every member's name in the README, and have each member open the repository and project brief using their own account before the team presents the documents.
