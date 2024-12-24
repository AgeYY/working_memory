# Beyond the Delay Neural Dynamics: a Decoding Strategy for Working Memory Error Reduction


## Introduction
This repository hosts the code accompanying the paper "Beyond the Delay Neural Dynamics: A Decoding Strategy for Working Memory Error Reduction," where we invesgated the influence of the decoding phase in reducing memory errors through the lens of artificial recurrent neural networks (RNNs).

TODO: Some codes in this repository are redundant and not relevant to the paper.

## Repository Structure

- **Results Reproducing**: refer to `./bin/main_figures.ipynb` for codes reproducing the main figures presented in the paper.
- **Pretrained Models**: Accessible [here](https://wustl.box.com/s/6qvrb1giv5ykzwetllhgg5uv9w4bploo). Please place the downloaded folder under `./core/`.
- **Figure Data**: To visualize figures without generating new data (which may take hours), locate the `fig_data` folder under `./bin/figs/`. Modify the `--gen_data` option in `./bin/main_figures.ipynb` to `False`, then execute the notebook.

## Environment Setup

To run this project, you need a Python environment with specific libraries for scientific computing and neural network modeling:

    - Python 3.8 or higher
    - PyTorch 1.10.1 with CUDA 11.3
    - matplotlib (for generating plots)
    - colormath 3.0.0
    - mpi4py 3.1.3
    - numpy, pandas, seaborn, scikit-learn, statsmodels, numba
    - Jupyter Notebook (optional for running experiment notebooks)

We also provided `working_memory_enviroment.yml` for reference. You can create a new conda environment with the provided `.yml` file:

```bash
conda env create -f working_memory_enviroment.yml
```

## Model Architecture

The RNN model used in our experiments is configured with the following parameters:

- Input Neurons: 12 (Perception neurons) + 1 (Go neuron)
- Output Neurons: 12 (Response neurons)
- Recurrent Neurons: 256
- Noise Strength: σ = 0.2 for intrinsic recurrent noise
- Tuning Curves: von Mises functions with shifting mean positions for input perception
- Delay Epoch Length: Randomly determined from a uniform distribution between 0 ms to 1000 ms in each trial
- Training Environment: Color input sampled from a prior distribution with four peaks indicating common colors more likely to occur

For more detailed parameters and their explanations, please refer to our manuscript.


## Running the Experiment

To replicate our experiments or test the model with different configurations, follow these steps:

- ### Model Training: ###
Run the training script to train the RNN model on the color delayed-response task. You can adjust the model parameters and training settings in the script.
```bash
python ./bin/train_cluster.py
```
which also allows parallel training
```bash
mpiexec -n n_model python ./bin/train_cluster.py
```

- ### Figure reproducing: ###
For a deeper analysis of the neural dynamics and decoding strategy, refer to the commands provided in `./bin/main_figures.ipynb`. These notebooks contain detailed visualizations and explanations of the results.
  

## Citation ##
If our research supports or inspires your work, please consider citing it as follows:
``` bash
@article{ye2023beyond,
  title={Beyond the Delay Neural Dynamics: A Decoding Strategy for Working Memory Error Reduction},
  author={Zeyuan Ye and Haoran Li and Liang Tian and Changsong Zhou},
  year={2023},
  note={Preprint on BioRxiv},
  url={https://www.biorxiv.org/content/10.1101/2022.06.01.494426v3}
}
```

## Acknowledgement ##
The code is based on the code of [Bi and Zhou, 2020](https://github.com/zedongbi/IntervalTiming).

1. Bi, Z. & Zhou, C. Understanding the computation of time using neural network models. Proc. Natl. Acad. Sci. U. S. A. 117, 10530–10540 (2020).
