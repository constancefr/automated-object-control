# Linear Cruise Control in Relative Coordinates
This is an adaptation of S. Teuber's Gymnasium implementation for VerSAILLE & Mosaic.
The environment consists of a single straight lane with:
- a leader car, which accelerates, brakes or idles at random,
- an ego car, for the purpose of controlling it to avoid collision and out-of-boundary escape.

## Installation
To install this environment, run the following commands:

```{shell}
cd versaille_env
pip install -e .
```
