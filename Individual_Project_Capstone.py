#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt


# ## Load Dataset

# In[2]:


df_train = pd.read_excel("Individual_Project.xlsx", sheet_name="Training")
df_test = pd.read_excel("Individual_Project.xlsx", sheet_name="Testing")


# ##  Data cleaning

# In[3]:


df_train.columns = ['Part #', 'FillRT', 'MT4', 'Y_True']
df_train[['FillRT', 'MT4', 'Y_True']] = df_train[['FillRT', 'MT4', 'Y_True']].apply(pd.to_numeric, errors='coerce')
df_train.dropna(inplace=True)

df_test.columns = ['Part #', 'FillRT', 'MT4', 'Y_True']
df_test[['FillRT', 'MT4', 'Y_True']] = df_test[['FillRT', 'MT4', 'Y_True']].apply(pd.to_numeric, errors='coerce')
df_test.dropna(inplace=True)


# In[4]:


fillrt = ctrl.Antecedent(np.arange(df_train['FillRT'].min(), df_train['FillRT'].max(), 0.001), 'FillRT')
mt4 = ctrl.Antecedent(np.arange(df_train['MT4'].min(), df_train['MT4'].max(), 0.1), 'MT4')
y_pred = ctrl.Consequent(np.arange(df_train['Y_True'].min(), df_train['Y_True'].max(), 0.1), 'Y_Pred')


# ## Membership Function

# In[5]:


fillrt['low'] = fuzz.trimf(fillrt.universe, [df_train['FillRT'].min(), df_train['FillRT'].min(), df_train['FillRT'].mean()])
fillrt['medium'] = fuzz.trimf(fillrt.universe, [df_train['FillRT'].min(), df_train['FillRT'].mean(), df_train['FillRT'].max()])
fillrt['high'] = fuzz.trimf(fillrt.universe, [df_train['FillRT'].mean(), df_train['FillRT'].max(), df_train['FillRT'].max()])

mt4['low'] = fuzz.trimf(mt4.universe, [df_train['MT4'].min(), df_train['MT4'].min(), df_train['MT4'].mean()])
mt4['medium'] = fuzz.trimf(mt4.universe, [df_train['MT4'].min(), df_train['MT4'].mean(), df_train['MT4'].max()])
mt4['high'] = fuzz.trimf(mt4.universe, [df_train['MT4'].mean(), df_train['MT4'].max(), df_train['MT4'].max()])

y_pred['low'] = fuzz.trimf(y_pred.universe, [df_train['Y_True'].min(), df_train['Y_True'].min(), df_train['Y_True'].mean()])
y_pred['medium'] = fuzz.trimf(y_pred.universe, [df_train['Y_True'].min(), df_train['Y_True'].mean(), df_train['Y_True'].max()])
y_pred['high'] = fuzz.trimf(y_pred.universe, [df_train['Y_True'].mean(), df_train['Y_True'].max(), df_train['Y_True'].max()])


# ## Fuzzy Rule Bank

# In[6]:


rules = [
    ctrl.Rule(fillrt['low'] & mt4['low'], y_pred['low']),
    ctrl.Rule(fillrt['low'] & mt4['medium'], y_pred['medium']),
    ctrl.Rule(fillrt['low'] & mt4['high'], y_pred['high']),
    ctrl.Rule(fillrt['medium'] & mt4['low'], y_pred['medium']),
    ctrl.Rule(fillrt['medium'] & mt4['medium'], y_pred['medium']),
    ctrl.Rule(fillrt['medium'] & mt4['high'], y_pred['high']),
    ctrl.Rule(fillrt['high'] & mt4['low'], y_pred['medium']),
    ctrl.Rule(fillrt['high'] & mt4['medium'], y_pred['high']),
    ctrl.Rule(fillrt['high'] & mt4['high'], y_pred['high'])
]


# In[7]:


prediction_ctrl = ctrl.ControlSystem(rules)
prediction_sim = ctrl.ControlSystemSimulation(prediction_ctrl)


# ## Compute Predictions 

# In[8]:


def compute_predictions(df_input):
    predictions = []
    for index in range(len(df_input)):
        sample_fillrt = df_input['FillRT'].iloc[index]
        sample_mt4 = df_input['MT4'].iloc[index]
        
        prediction_sim.input['FillRT'] = sample_fillrt
        prediction_sim.input['MT4'] = sample_mt4
        
        prediction_sim.compute()
        predictions.append(prediction_sim.output['Y_Pred'])
    return predictions


# In[9]:


train_predictions = compute_predictions(df_train)
test_predictions = compute_predictions(df_test)


# ## Predictions and Model Accuracy

# In[10]:


df_train['Y_Pred'] = train_predictions
df_train['Accuracy'] = 100 - ((abs(df_train['Y_Pred'] - df_train['Y_True']) / df_train['Y_True']) * 100)

df_test['Y_Pred'] = test_predictions
df_test['Accuracy'] = 100 - ((abs(df_test['Y_Pred'] - df_test['Y_True']) / df_test['Y_True']) * 100)

with pd.ExcelWriter("Predicted_Y_Results_Split.xlsx") as writer:
    df_train.to_excel(writer, sheet_name='Training Set', index=False)
    df_test.to_excel(writer, sheet_name='Testing Set', index=False)


# In[11]:


with pd.ExcelWriter("Predicted_Y_Results_Split.xlsx") as writer:
    df_train.to_excel(writer, sheet_name='Training Set', index=False)
    df_test.to_excel(writer, sheet_name='Testing Set', index=False)


# In[12]:


train_accuracy = df_train['Accuracy'].mean()
test_accuracy = df_test['Accuracy'].mean()

print("\n" + "-"*30)
print("MODEL ACCURACY")
print("-"*30)
print(f"Training Accuracy: {train_accuracy:.3f}%")
print(f"Testing Accuracy: {test_accuracy:.3f}%")


# ## Average Accuracy

# In[13]:


avg_accuracy = (df_train['Accuracy'].mean() + df_test['Accuracy'].mean()) / 2


# In[14]:


print("\n" + "-"*30)
print("AVERAGE ACCURACY")
print("-"*30)
print(f"avg_accuracy: {avg_accuracy:.5f}%")


# ## MF Visualization
# 

# In[15]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  


axes[0].plot(fillrt.universe, fuzz.trimf(fillrt.universe, [df_train['FillRT'].min(), df_train['FillRT'].min(), df_train['FillRT'].mean()]), label="Low")
axes[0].plot(fillrt.universe, fuzz.trimf(fillrt.universe, [df_train['FillRT'].min(), df_train['FillRT'].mean(), df_train['FillRT'].max()]), label="Medium")
axes[0].plot(fillrt.universe, fuzz.trimf(fillrt.universe, [df_train['FillRT'].mean(), df_train['FillRT'].max(), df_train['FillRT'].max()]), label="High")
axes[0].set_title("FillRT Membership Function")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(mt4.universe, fuzz.trimf(mt4.universe, [df_train['MT4'].min(), df_train['MT4'].min(), df_train['MT4'].mean()]), label="Low")
axes[1].plot(mt4.universe, fuzz.trimf(mt4.universe, [df_train['MT4'].min(), df_train['MT4'].mean(), df_train['MT4'].max()]), label="Medium")
axes[1].plot(mt4.universe, fuzz.trimf(mt4.universe, [df_train['MT4'].mean(), df_train['MT4'].max(), df_train['MT4'].max()]), label="High")
axes[1].set_title("MT4 Membership Function")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(y_pred.universe, fuzz.trimf(y_pred.universe, [df_train['Y_True'].min(), df_train['Y_True'].min(), df_train['Y_True'].mean()]), label="Low")
axes[2].plot(y_pred.universe, fuzz.trimf(y_pred.universe, [df_train['Y_True'].min(), df_train['Y_True'].mean(), df_train['Y_True'].max()]), label="Medium")
axes[2].plot(y_pred.universe, fuzz.trimf(y_pred.universe, [df_train['Y_True'].mean(), df_train['Y_True'].max(), df_train['Y_True'].max()]), label="High")
axes[2].set_title("Y_Pred Membership Function")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


# In[ ]:




