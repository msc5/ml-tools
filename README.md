# ml-tools

By Alkin Kaz, Sam Liang, Matthew Coleman

## Learning-Task-Agnostic, Performant, Simple, Dataset Sampling

Datasets often follow some heirarchical structure, with nested folders in a tree-like data structure. By specifying *how* to sample images at each level of the tree, ml-tools allows you to spend less time worrying about your dataset configuration and more time working on your models.

### Easy Batch Sampling

For example, say you needed to batch two images for each "episode" of training:

<p align='center'><img src="vis/batch.gif" width=400/><\p>
  
### Few-Shot Sampling

Or say you needed k batches of n + m sub-batches, taken from specific levels of the directory tree for each episode, creating a few-shot-esque task:

<p align='center'><img src="vis/fewshot.gif" width=400/><\p>
