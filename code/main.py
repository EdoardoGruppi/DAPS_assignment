# Import packages
import tensorflow as tf

# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ACQUISITION ==========================================================================================================
# todo put others section

# BASE =================================================================================================================
# training_batches, valid_batches, test_batches = data_preprocessing(...)
# input_shape = ...
# # Build model object.
# model= base(input_shape)
# # Train model based on the training set
# base_train, base_valid = model.train(...)
# # Test model based on the test set.
# base_test = model.test(...)
# # Clean up memory
# del ...

# ADV ==================================================================================================================
# training_batches, valid_batches, test_batches = data_preprocessing(...)
# input_shape = ...
# # Build model object.
# model= base(input_shape)
# # Train model based on the training set
# base_train, base_valid = model.train(...)
# # Test model based on the test set.
# base_test = model.test(...)
# # Clean up memory
# del ...

# RESULTS ==============================================================================================================
# Print out your results with following format:
print('Model  {:<12} {:<12} {:<12} {:<12}\n'.format('Train Acc', 'Valid Acc', 'Test Acc', 'Test 2 Acc'),
      'Base:  {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format(0, 1, 2, 3),
      'Adv:  {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format(0, 1, 2, 3))
