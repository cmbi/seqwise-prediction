# Sequence-based protein class prediction.

## Dependencies
 - python 3.11.0
 - scikit-learn 1.4.2 
 - h5py 3.11.0 
 - pytorch 1.0.2 

## Data Format
The input data must be presented as an hdf5 file, containing the following:
 - peptide/sequence_onehot: a 9x22 array, containing one-hot encoded amino acid sequence
 - affinity: a value between 0.0 (low affinity) to 1.0 (high affinity)

## Training & testing a flattening model
```
python run.py flattening train.hdf5 valid.hdf5 test.hdf5 results.csv
```
