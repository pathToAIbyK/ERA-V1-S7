# ERA-V1-S7

## Code-1 - https://github.com/pathToAIbyK/ERA-V1-S7/blob/main/S7_model1.ipynb
  ### Target
    -Get the set-up right
    -Set Transforms
    -Set Data Loader
    -Set Basic Working Code
    -Set Basic Training  & Test Loop
  ### Results:
    -Parameters: 2,314
    -Best Training Accuracy: 11.24
    -Best Test Accuracy: 11.35
  ### Analysis:
    -Very bad model
    -Model is not learning and jsut mimicing!
  
## Code-2 - https://github.com/pathToAIbyK/ERA-V1-S7/blob/main/S7_model2.ipynb
  ### Target
    -Get the Skeleton right
    -Increase the number of parameters but < 8000
    -Introduce Squeeze and expand of channels
  ### Results:
    -Parameters: 6050
    -Best Training Accuracy: 99.11 (15th Epooch)
    -Best Test Accuracy: 99.04 (14th Epooch)
  ### Analysis:
    -Good model! the difference between train and test is very less
    -The model is underfitting 
