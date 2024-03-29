# TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization
Pytorch implementation for TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization.

# Notation
Compared with the original implementation, we add skip connection between encoder and decoder, which can improve the performance.

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Pytorch >= 1.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train'
```

- inference
```
python3 main.py --mode 'inf'
```

## Citation 
```
@article{Han-VIS19,
	Author = {J. Han and C. Wang},
	Journal = {IEEE Transactions on Visualization and Computer Graphics},
	Number = {1},
	Pages = {205-215},
	Title = {{TSR-TVD}: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization},
	Volume = {26},
	Year = {2020}}

```
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1455886, CNS-1629914, and DUE-1833129.
