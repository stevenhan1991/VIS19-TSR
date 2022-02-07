# TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization
Pytorch implementation for TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.6
- Numpy
- Pytorch >= 0.4.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd code 
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
