clear all; clc; clear;

%% Make FileDatastores

jonah_path = "/run/user/1000/gvfs/smb-share:server=ripoff.math.colostate.edu,share=eit/Anatomical_Atlas_3D/Babies_GE";

[train_ds, val_ds, test_ds] = makeDatastores(jonah_path, 0.75, 0.15);

save("datastores.mat", "train_ds", "val_ds", "test_ds");

%% Make ArrayDatastores

load("datastores.mat", "train_ds", "val_ds", "test_ds");

[train_ds, val_ds, test_ds] = makeArrayDatastores(train_ds, val_ds, test_ds);

save("arraydatastores.mat", "train_ds", "val_ds", "test_ds");

