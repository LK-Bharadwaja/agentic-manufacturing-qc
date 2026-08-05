# +
import pandas as pd
import asyncio
import xlsxwriter 
from openpyxl import load_workbook
from openpyxl import Workbook
import os

async def rules(x5_key, x6_key, x49_key, x54_key, y_vals):   
    if x5_key == 's' and x6_key == 's' and x49_key == 's' and x54_key == 'h':
        return y_vals.get("min_diameter", 0)
    else:
        return y_vals.get("avg_diameter", 0)

async def calculateMU(min_value, max_value, avg_value, row_value):
    mu_values = {}
    if row_value >= min_value and row_value <= avg_value:
        centroid = (avg_value + min_value) / 2
        mu_1 = (row_value - min_value) / (avg_value - min_value)
        mu_2 = (avg_value - row_value) / (avg_value - min_value)
        if row_value > centroid:
            mu_values["s"] = min(mu_1, mu_2)
            mu_values["m"] = max(mu_1, mu_2)
        elif row_value < centroid:
            mu_values["s"] = max(mu_1, mu_2)
            mu_values["m"] = min(mu_1, mu_2)
        else:
            mu_values["s"] = 0.5
            mu_values["m"] = 0.5
    if row_value >= avg_value and row_value <= max_value:
        centroid = (max_value + avg_value) / 2
        mu_1 = (row_value - avg_value) / (max_value - avg_value)
        mu_2 = (max_value - row_value) / (max_value - avg_value)
        if row_value > centroid:
            mu_values["m"] = min(mu_1, mu_2)
            mu_values["h"] = max(mu_1, mu_2)
        elif row_value < centroid:
            mu_values["m"] = max(mu_1, mu_2)
            mu_values["h"] = min(mu_1, mu_2)
        else:
            mu_values["m"] = 0.5
            mu_values["h"] = 0.5
    return mu_values

async def calculateAccuracy():
    try:
        df = pd.read_excel("IME _670_capstone-stage-2.xlsx")
        x5_values = {"min_x5": df["x5"].min(), "max_x5": df["x5"].max(), "avg_x5": df["x5"].mean()}
        x6_values = {"min_x6": df["x6"].min(), "max_x6": df["x6"].max(), "avg_x6": df["x6"].mean()}
        x49_values = {"min_x49": df["x49"].min(), "max_x49": df["x49"].max(), "avg_x49": df["x49"].mean()}
        x54_values = {"min_x54": df["x54"].min(), "max_x54": df["x54"].max(), "avg_x54": df["x54"].mean()}
        diameter_values = {"min_diameter": df["Diameter"].min(), "max_diameter": df["Diameter"].max(), "avg_diameter": df["Diameter"].mean()}
        accuracy_list = []
        prediction_list = []
        for p in df.index:
            row_values = {"x5": df["x5"][p], "x6": df["x6"][p], "x49": df["x49"][p], "x54": df["x54"][p], "diameter": df["Diameter"][p]}
            x5_mu_values = await calculateMU(x5_values["min_x5"], x5_values["max_x5"], x5_values["avg_x5"], row_values["x5"])
            x6_mu_values = await calculateMU(x6_values["min_x6"], x6_values["max_x6"], x6_values["avg_x6"], row_values["x6"])
            x49_mu_values = await calculateMU(x49_values["min_x49"], x49_values["max_x49"], x49_values["avg_x49"], row_values["x49"])
            x54_mu_values = await calculateMU(x54_values["min_x54"], x54_values["max_x54"], x54_values["avg_x54"], row_values["x54"])
            # diameter_mu_values = await calculateMU(diameter_values["min_diameter"], diameter_values["max_diameter"], diameter_values["avg_diameter"], row_values["diameter"])
            y_pred = 0
            for x5_key, x5_val in x5_mu_values.items():
                for x6_key, x6_val in x6_mu_values.items():
                    for x49_key, x49_val in x49_mu_values.items():
                        for x54_key, x54_val in x54_mu_values.items():
                            y_1 = await rules(x5_key, x6_key, x49_key, x54_key, diameter_values)
                            
                            if y_1:
                                y_pred += x5_val * x6_val * x49_val * x54_val * y_1
                                
            prediction_list.append(y_pred)
            accuracy = 100 - ((abs(y_pred - row_values["diameter"]) / row_values["diameter"]) * 100)
            accuracy_list.append(accuracy)
        
        df["y_prediction"] = prediction_list
        df["accuracy"] = accuracy_list
        df.to_excel("group-project-accuracy.xlsx", index=False)
        average_accuracy = df["accuracy"].mean()
        print(f"Successfully calculated the average accuracy is {average_accuracy}")
        return f"Successfully calculated the average accuracy is {average_accuracy}"
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

asyncio.run(calculateAccuracy())



