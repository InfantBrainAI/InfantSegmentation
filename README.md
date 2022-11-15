# CloudInfant

## Usage:

### training
```
python train.py --name YOURNAME --checkpoints_dir YOURPATH --dataroot YOURPATH
```

find more configurations in [**/options/base_options.py**](/options/base_options.py) and [**/options/train_options.py**](/options/train_options.py)


### testing 
```
python test.py --name YOURNAME --checkpoints_dir YOURPATH --dataroot YOURPATH --whichmodel YOURMODELNAME 
```

find more configurations in [**/options/base_options.py**](/options/base_options.py) and [**/options/test_options.py**](/options/test_options.py)
