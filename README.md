# Convolutional-Linear-Neural-Network-2-12
Neural Net using PyTorch with convolutional and linear layers based off of LeNet-5. Hardcoded to access the MNIST database.

Ensure PyTorch and Numpy are installed
Uses PyTorch to access the data interpret it
If using MNIST to train handwritten digits, use hardcoded Datasets (see lines 22 and 23 of DigitRecognizer.py)
Otherwise modify as desired.

change epochs variable to adjust how many training epochs being run through (currently set to 5)

Reminder: gradient descent needs to be rewritten to iterate only after batches, not each datums
Saved network trained to ~.97 accuracy, but run times could be improved
