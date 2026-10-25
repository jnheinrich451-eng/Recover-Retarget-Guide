# Development notebooks

These four notebooks are the Colab sessions in which the thesis's runs were made, kept as the record of how the experiment
was driven: `colab.ipynb` (Modules 1 to 3 on the development set), `colab_experiment.ipynb` (the test set),
`colab_ablation_guidance.ipynb` (the guidance ablations) and `colab_m2_selection.ipynb` (the anchor selection report).
Their outputs are cleared.

They are bound to the author's set-up: they clone the private development repository with a token read from the author's
Google Drive, and they read and write the author's Drive folders (`MyDrive/motion_transfer`, `MyDrive/motion_transfer_exp`).
They are not the entry point for running the code. For that, use `colab_quickstart.ipynb` at the root of the repository, or
the runners described in the README with `MT_ROOT` pointing at your own data folder.
