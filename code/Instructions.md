# Instructions

## Setup

1. Install Tensorflow and all the other packages appointed in the [README.md](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/README.md) file.
2. Install Alpha Advantage package....
```
pip install alpha_vantage
```
3. Install twint
```
pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
```
3. Install flair. to run with GPU you must have torch with cuda enabled
```python 
import torch
import flair
device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)
torch.zeros(1).cuda()
```
```
pip install flair
```
## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal.

**Note:** To follow step by step the main execution take a look on the dedicated Section below.

```
python main.py
```

## Datasets

### Dataset division

## Model (or models)

## Main execution

## Issues
