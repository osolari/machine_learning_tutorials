# ML From Scratch

Common Machine Learning algorithms/Models implementation from scratch.

## Algorithms Implemented

- [Supervised Learning](https://github.com/osolari/ML_from_scratch/tree/main/src/supervised)
  - [KNN](https://github.com/osolari/ML_from_scratch/blob/main/src/supervised/knn.py)
- [Unsupervised Learning](https://github.com/osolari/ML_from_scratch/tree/main/src/unsupervised)
  - [Kmeans](https://github.com/osolari/ML_from_scratch/blob/main/src/unsupervised/kmeans.py)
  - [PCA](https://github.com/osolari/ML_from_scratch/blob/main/src/unsupervised/pca.py)

## Installation and usage.

This project has 2 dependencies.

You can install these using the command below!

```sh
pip install -r requirements.txt
```

You can run the files as following.

```sh
python -m mlfromscratch.<algorithm-file>
```

with `<algorithm-file>` being the valid filename of the algorithm without the extension.

For example, If I want to run the Linear regression example, I would do 
`python -m mlfromscratch.linear_regression`