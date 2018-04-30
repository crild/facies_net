# facies_net
Software for detection of given facies

Functions needed to run facies_net.py are contained in the folder facies_netfunc, logs hold TensorBoard data, and F3 holds trained models.

The user needs the F3 dataset in the same folder as facies_net.py, as well as classification in .pts files.

The file will save training results in TensorBoard-format, simply go into the terminal and write: tensorboard --logdir=logs/"name-of-folder" and then open a new web browser and write "localhost:6006" in the adress bar.

e.g. to view the results from F3_train write: tensorboard --logdir=logs/F3 then open a new tab in chrome and input localhost:6006
