import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/palmer-penguins/palmer-penguins.csv"
df = pd.read_csv(url)

''' 
Use a pandas summary table (df.groupby(...).aggregate(...)) to answer the following question: how does the mean mass of penguins vary by species and sex? 
''' 

table = df.groupby(['Species', 'Sex']).aggregate({'Body Mass (g)' : 'mean'})

print(table)

'''
Make a scatterplot of culmen length against culmen depth, with the color of each point corresponding to the penguin species.
'''

sns.scatterplot(data = df, x = 'Culmen Length (mm)', y ='Culmen Depth (mm)', hue='Species')
