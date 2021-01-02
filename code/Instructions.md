# Instructions

## Setup

1. Install all the other packages appointed in the [README.md](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/README.md) file using conda or pip.
2. To install Twint could be necessary to run the code below to be sure that the last version is installed.
   ```
   pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
   ```
3. Install flair. Note that to run Flair with GPU it is required to have torch with cuda enabled.

   ```
   pip install flair
   ```

   Optional:

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

4. When installing fbprophet it is needed to run the following lines since there is a conflict between specific version of fbprophet and pystan.
   ```
   conda install -c conda-forge fbprophet
   conda install -c conda-forge pystan
   ```

## Run the code

ALTRO

Once all the necessary packages have been installed you can run the code by typing this line on the terminal.

```
python main.py
```

## Additional information

A more detailed overview of the project is provided in the report published (the pdf file).

## Issues
