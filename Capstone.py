# importing packages
import sys
import tkinter
from tkinter import filedialog, messagebox, Label, Text, Button
import pandas as pd
import numpy as np
import pickle
import sklearn
import math
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from fiona import schema
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier

shapes = "zipShapes/oklahoma-zip-code-boundaries.shp"
shapes_path = os.path.join(os.getcwd(), "zipShapes", "oklahoma-zip-code-boundaries.shp")
demo_path = os.path.join(os.getcwd(), "DemographicData.csv")
model_path = os.path.join(os.getcwd(), "finalModel.pkl")
original_dir = os.getcwd()

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

m = tkinter.Tk()
m.title("Krish-Capstone")
m.geometry("900x500")


# needed to maintain cwd!

#############
# HANDLERS #
#############
def classifier_handler(event=None):
    filepath = ""
    try:
        year = int(inputtxt.get(1.0, "end-1c"))
    except:
        messagebox.showerror("showerror", "Input a year")
        return

    if year < 2018 or year > 3000:
        messagebox.showerror("showerror", "Input a valid year")
        return

    while ".csv" not in filepath:
        # resetting file path everytime, incase they select wrong file

        filepath = ""
        filepath = filedialog.askopenfilename()
        # kills the loop if they close the pop-up!
        if filepath == "":
            return
    # no years less than 2018, no years greater than 3000
    results = preform_classification(filepath, year)
    if type(results) == str:
        messagebox.showerror("showerror", results)
        return
    else:
        download_directory = filedialog.askdirectory(initialdir="YOUR DIRECTORY PATH",
                                                     title="Where Do I Save Classification Results?")
        results.to_csv(f"{download_directory}/Classification_{year}.csv")


def regression_handler(event=None):
    filepath = ""
    try:
        year = int(inputtxt.get(1.0, "end-1c"))
    except:
        messagebox.showerror("showerror", "Input a year")
        return

    if year < 2018 or year > 3000:
        messagebox.showerror("showerror", "Input a valid year")
        return

    while ".csv" not in filepath:
        # resetting file path everytime, incase they select wrong file

        filepath = ""
        filepath = filedialog.askopenfilename()
        # kills the loop if they close the pop-up!
        if filepath == "":
            return

    frequency_dict, df = convert_to_df(filepath)
    if type(df) == str:
        messagebox.showerror("showerror", "Invalid Input")
        return
    results, name = regression_analysis(df, year)
    if type(results) == str:
        messagebox.showerror("showerror", results)
        return

    for key in frequency_dict.keys():
        results[f"{key} PREDICTION"] = (results[f"PREDICTION {year}"] * frequency_dict[key])

    download_directory = filedialog.askdirectory(initialdir="YOUR DIRECTORY PATH",
                                                 title="Where Do I Save Prediction Results?")
    results.to_csv(f"{download_directory}/RAW_Prediction_{year}.csv")
    easy_results = round(results.sum())
    easy_results.rename("Equipment Count", inplace=True)
    easy_results.to_csv(f"{download_directory}/FINAL_Prediction_{year}.csv")


def map_handler(event=None):
    filepath = ""
    try:
        year_list = process_year(inputtxt.get(1.0, "end-1c"))
        text_year = inputtxt.get(1.0, "end-1c")
    except:
        messagebox.showerror("showerror", "Input a proper range (eg. 2018-2022) or year")
        return

    if type(year_list) == str:
        messagebox.showerror("showerror", "Input a proper range (eg. 2018-2022) or year")
        return

    while ".csv" not in filepath:
        # resetting file path everytime, incase they select wrong file

        filepath = ""
        filepath = filedialog.askopenfilename()
        # kills the loop if they close the pop-up!
        if filepath == "":
            return

    _, df = convert_to_df(filepath)
    if type(df) == str:
        messagebox.showerror("showerror", "Improper Input")
        return

    df = df[df["DATE"].isin(year_list)]
    df.drop("DATE", axis=1, inplace=True)
    df["ZIP CODE"] = df["ZIP CODE"].astype(int).astype(str)

    try:
        zip_shapes = gpd.read_file("zipShapes/oklahoma-zip-code-boundaries.shp")
    except:
        messagebox.showerror("showerror",
                             f"Missing Zip Shapes file {os.path.dirname(os.path.abspath(__file__))}, {shapes_path}")
        return

    zip_shapes["zcta5ce00"] = zip_shapes["zcta5ce00"].astype(str)
    non_code = (zip_shapes["zcta5ce00"][~zip_shapes["zcta5ce00"].isin(df["ZIP CODE"])]).values
    filled_df = pd.DataFrame(list(zip(non_code, np.zeros(len(non_code)))), columns=["ZIP CODE", "Given"])
    df = pd.concat([df, filled_df], axis=0)
    df = df.groupby(["ZIP CODE"])["Given"].apply(np.sum, axis=0).reset_index()
    df["Adjusted"] = df["Given"].map(given_map)

    try:
        heatmap_df = zip_shapes.merge(df, right_on="ZIP CODE", left_on="zcta5ce00")
    except:
        messagebox.showerror("showerror", "Zip Shapes File ERROR")
        return

    # making the figure!
    fig, ax = plt.subplots(figsize=(9, 4))
    heatmap_df.plot(legend=True, column="Adjusted", legend_kwds={'loc': 'lower left',
                                                         'bbox_to_anchor': (0, 0.1),
                                                         'markerscale': 1.29,
                                                         'title_fontsize': 'medium',
                                                         'fontsize': 'small'}, cmap="plasma", ax=ax)

    leg1 = ax.get_legend()
    leg1.set_title("Equipment Donated")
    new_legtxt = ["0", "<10", "<50", "<200", "<1000", ">1000"]
    for i, eb in enumerate(leg1.get_texts()):
        eb.set_text(new_legtxt[i])

    plt.gca().axis('off')
    plt.title(f"Cleats 4 Kids Donation Heatmap for {text_year}")

    download_directory = filedialog.askdirectory(initialdir="YOUR DIRECTORY PATH",
                                                 title="Where Do I Put Figures?")
    plt.savefig(f"{download_directory}/Heatmap_Zipcode_{text_year}.png")



    zip_shapes = gpd.read_file("county/COUNTY_BOUNDARY.shp")
    counties_df = pd.read_csv("counties.csv")
    counties_df["ZIP CODE"] = counties_df["ZIP CODE"].astype(float)
    counties_df.set_index("ZIP CODE", inplace=True)
    # some renaming
    counties_map = df
    counties_map["ZIP CODE"]= pd.to_numeric(counties_map["ZIP CODE"], errors = "coerce")
    counties_map.dropna(axis = 0, inplace=True)
    counties_map.set_index("ZIP CODE", inplace=True)
    counties_final = pd.merge(counties_map, counties_df, right_index=True, left_index=True)
    counties_count = counties_final.groupby(["County"])["Given"].apply(np.sum, axis=0).reset_index()
    counties_count.set_index("County", inplace=True)
    counties_count = counties_count.reset_index()
    counties_count["County"] = (counties_count["County"]).map(county_convert)
    counties_count["Adjusted"] = counties_count["Given"].map(given_map3)
    other_df = zip_shapes.merge(counties_count, right_on="County", left_on="COUNTY_NAM", how="left")
    other_df["Adjusted"] = other_df["Adjusted"].replace(np.nan, 0)

    fig, ax = plt.subplots(figsize=(9, 4))
    other_df.plot(legend=True, column="Adjusted", legend_kwds={'loc': 'lower left',
                                                               'bbox_to_anchor': (0, 0.1),
                                                               'markerscale': 1.29,
                                                               'title_fontsize': 'medium',
                                                               'fontsize': 'small'}, cmap="plasma", ax=ax)

    leg1 = ax.get_legend()
    leg1.set_title("Items Donated")
    new_legtxt = ["0", "<100", "<500", "<1000", "<2500", ">2500"]
    for i, eb in enumerate(leg1.get_texts()):
        eb.set_text(new_legtxt[i])

    plt.gca().axis('off')
    plt.title(f"Cleats 4 Kids Donation Heatmap for {text_year}")
    plt.savefig(f"{download_directory}/Heatmap_County_{text_year}.png")


#######################################
# CRITICAL FUNCTIONS USED BY HANDLERS #
#######################################
def preform_classification(filepath, year):
    try:
        df = pd.read_csv(filepath)
        df.columns = np.array(["ZIPCODES"])
        df.set_index("ZIPCODES", inplace=True)
    except:
        return "Improper input"

    try:
        #zip_df = pd.read_csv(resource_path("DemographicData.csv"))

        # so the demographic path is changeable! This data must be included
        zip_df = pd.read_csv(demo_path)
    except Exception:
        return f"Demographics not available, {demo_path}"

    zip_df = zip_df.rename({"Your Entry": "ZIPCODES"}, axis=1).set_index("ZIPCODES")
    full_df = pd.merge(df, zip_df, right_index=True, left_index=True)
    # getting rid of columns with no values (NA)
    full_df.dropna(axis=0, inplace=True)
    # 2018 is the 0th year, after that it goes up!
    full_df["DATE"] = year - 2018
    corr_features = ['Population', 'Business Annual Payroll', 'Households',
                     'Asian Population', 'Black or African American Population',
                     'American Indian or Alaskan Native Population', 'White Population',
                     'DATE']

    # what the hell is the problem with this line?
    try:
        model = pd.read_pickle(resource_path("finalModel.pkl"))
        #model = pd.read_csv(model_path)
        '''
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        '''
    except:
        return f"Model not found, {resource_path("finalModel.pkl")}"

    full_df["Classification"] = np.array(map(make_readable, model.predict(full_df[corr_features])))
    return full_df["Classification"]


def regression_analysis(df, year):
    # valid zipcodes only check, and filling in other Oklahoma Zipcodes
    try:
        zip_shapes = gpd.read_file(f"{original_dir}/zipShapes/oklahoma-zip-code-boundaries.shp")
    except:
        return "Missing Boundaries", None
    float_items = []
    for item in zip_shapes["zcta5ce00"]:
        try:
            float_items.append(float(item))
        except:
            pass
    non_code = [element for element in float_items if element not in np.array(df["ZIP CODE"])]

    # this just makes it dynamic
    date_array = []
    for number in np.unique(df["DATE"]):
        date_array += list(np.full(len(non_code), int(number)))

    filled_df = pd.DataFrame(list(
        zip(np.tile(non_code, len(np.unique(df["DATE"]))), np.zeros(len(non_code) * len(np.unique(df["DATE"]))),
            date_array)), columns=["ZIP CODE", "Given", "DATE"])

    df = pd.concat([filled_df, df], axis=0)
    df = df.set_index("ZIP CODE")
    df.index = np.array(df.index).astype('U5').astype(str)
    df.index.name = "ZIP CODE"
    # getting rid of the irrelevant info
    df = df[["DATE", "Given"]]
    df["DATE"] = df["DATE"] - 2018
    combined_df = None

    # creating a new type of dataframe, zipcodes are index, items per year are columns, empty years are filled with zero
    for index in list(set(df.index)):
        test_df = df.loc[index]
        # manipulate these areas, rotate on DATE

        if type(test_df) == pd.core.series.Series:
            test_df = pd.DataFrame(test_df).transpose()

        placeholder_dict = {}
        # goes through all years in the df, if empty makes values 0
        for j in range(max(df["DATE"]) + 1):
            if len(test_df.loc[test_df["DATE"] == j]) > 1:
                placeholder_dict[j] = [float(test_df.loc[test_df["DATE"] == j]["Given"].sum())]
            else:
                try:
                    placeholder_dict[j] = [(test_df.loc[test_df["DATE"] == j]["Given"].values)[0]]
                except:
                    placeholder_dict[j] = [0]
        final_df = pd.DataFrame.from_dict(placeholder_dict)
        final_df.index = [index]

        if combined_df is None:
            combined_df = final_df
        else:
            combined_df = pd.concat([combined_df, final_df], axis=0)
    del df
    # for the logarithm regression
    combined_df = combined_df.replace(0, .1)
    combined_df[f"PREDICTION {year}"] = combined_df.apply(run_equation, args=(year,), axis=1)

    # simple conversion
    column_dict = {}
    print(list(combined_df.keys()))
    for key, value in zip(list(combined_df.keys()), list(map(convert_key_to_year, combined_df.keys()))):
        column_dict[key] = value

    combined_df.rename(column_dict, axis=1, inplace=True)
    combined_df.replace(0.1, 0, inplace=True)
    return combined_df, f"Prediction for {year}"


##################
# MISC FUNCTIONS #
##################
def county_convert(county):
    return (county.rsplit(" ", 1)[0]).upper()

def given_map3(number):
    if number == 0:
        return "0"
    elif number < 100:
        return "1"
    elif number < 500:
        return "2"
    elif number < 1000:
        return "3"
    elif number < 2500:
        return "4"
    return "5"
def best_regression_model(K, Y):
    # Reshape X to be 2D
    reshape_X = K.reshape(-1, 1)
    # Initialize variables to store the best model info
    best_r2 = -np.inf
    best_equation = None

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(reshape_X, Y)
    y_pred_linear = linear_model.predict(reshape_X)
    r2_linear = r2_score(Y, y_pred_linear)
    if r2_linear > best_r2:
        best_r2 = r2_linear
        best_equation = (lambda x: (linear_model.coef_[0] * x + linear_model.intercept_), best_r2)

    # Quadratic Regression
    quadratic_transform = PolynomialFeatures(degree=2)
    X_quad = quadratic_transform.fit_transform(reshape_X)
    quad_model = LinearRegression()
    quad_model.fit(X_quad, Y)
    y_pred_quad = quad_model.predict(X_quad)
    r2_quad = r2_score(Y, y_pred_quad)
    if r2_quad > best_r2:
        best_r2 = r2_quad
        best_equation = (
            lambda x: (quad_model.coef_[2] * (x * x) + quad_model.coef_[1] * x + quad_model.intercept_), best_r2)

    # Exponential Regression (y = a * exp(b * x))
    Y_log = np.log(Y)
    exp_model = LinearRegression()
    exp_model.fit(reshape_X, Y_log)
    y_pred_exp = np.exp(exp_model.predict(reshape_X))
    r2_exp = r2_score(Y, y_pred_exp)
    if r2_exp > best_r2:
        best_r2 = r2_exp
        best_equation = (lambda x: (math.exp(exp_model.intercept_) * math.exp(exp_model.coef_[0] * x)), best_r2)

    # Logarithmic Regression (y = a * log(x) + b)
    X_log = np.log(K.astype(float) + 1e-6).reshape(-1, 1)
    X_log[0] = -100
    log_model = LinearRegression()
    log_model.fit(X_log, Y)
    y_pred_log = log_model.predict(X_log)
    r2_log = r2_score(Y, y_pred_log)
    if r2_log > best_r2:
        best_r2 = r2_log
        best_equation = (lambda x: (log_model.coef_[0] * math.log(x) + log_model.intercept_), best_r2)

    return best_equation


def run_equation(row, predict_year):
    predict_year = predict_year - 2018
    given_values = []
    # largest year
    for key in row.keys():
        given_values.append(row[key])

    equation, equation_score = best_regression_model(deepcopy(row.keys().values), deepcopy(np.array(given_values)))

    # if the r2 is less than 0.4 (bad fit) just take the average of the previous two values
    if equation_score > 0.4:
        return (max(round(equation(predict_year)), 0))
    else:
        # indexing using last 2 given, ignoring first
        return round(np.mean(np.array(given_values[-2:])))


def string_to_float(number):
    if type(number) is float or type(number) is int:
        return number
    if "," in number:
        number = number.replace(",", "")
        return float(number)
    else:
        return float(number)


def convert_key_to_year(item):
    try:
        year = int(item)
        return year + 2018
    except:
        return item


def convert_to_df(filepath):
    try:
        df = pd.read_csv(filepath)
    except:
        return None, "Improper input"
    mask = np.isin(np.array(["DATE", "ZIP CODE", "# OF NEW", "# OF GOOD", " # OF UA", "# OF PUR", "SPORT"]), df.keys())
    if False in mask:
        return None, "Missing Columns"

    # making the strings into floats
    df["# OF NEW"] = df["# OF NEW"].map(string_to_float)
    df["# OF GOOD"] = df["# OF GOOD"].map(string_to_float)
    df[" # OF UA"] = df[" # OF UA"].map(string_to_float)
    df["# OF PUR"] = df["# OF PUR"].map(string_to_float)

    # total given
    df["Given"] = df["# OF NEW"] + df["# OF GOOD"] + df[" # OF UA"] + df["# OF PUR"]
    # trimming dataframe

    translate_dict = {"RUN": "RUNNING", "BASETBALL": "BASKETBALL", "BASBALL": "BASEBALL", "UA BASEBALL": "BASEBALL",
                      "UA BB": "BASEBALL", "FB": "FOOTBALL", "UA FOOTBALL": "FOOTBALL", "UA FB CLEATS": "FOOTBALL",
                      "MISC.": "MISC", "NA": "MISC", "MISCELLANEOUS": "MISC", "VB": "VOLLEYBALL", "SB": "SOFTBALL",
                      "UA SB": "SOFTBALL", "lacross": "LACROSSE"}
    frequency = {"RUNNING": 0, "SOCCER": 0, "BASKETBALL": 0, "BASEBALL": 0, "FOOTBALL": 0, "MISC": 0, "TRACK": 0,
                 "VOLLEYBALL": 0, "DANCE": 0, "GOLF": 0, "SOFTBALL": 0, "WRESTLING": 0, "LACROSSE": 0, "BIKE": 0,
                 "TENNIS": 0, "CHEER": 0, "THUNDER": 0, "CLOTHING": 0, "GYM": 0, "SWIM": 0}

    sport_df = df[["SPORT", "Given"]].dropna()
    total_count = 0
    # dictionary with percentages
    for row in range(len(sport_df)):
        series = sport_df.iloc[row]
        sport = str(series["SPORT"]).strip().upper()
        if sport in translate_dict:
            sport = translate_dict[sport]
        if sport in frequency:
            frequency[sport] += float(series["Given"])
            total_count += float(series["Given"])

    for key in frequency.keys():
        frequency[key] = frequency[key] / total_count

    df = df[["DATE", "ZIP CODE", "Given"]]

    return (frequency, df.groupby(["DATE", "ZIP CODE"])["Given"].apply(np.sum, axis=0).reset_index())


def make_readable(item):
    if item == 0:
        return "<25 Items"
    elif item == 1:
        return ">25 Items"


def process_year(string):
    try:
        int(string)
        return [int(string)]
    except:
        pass

    if "-" not in string:
        return "ERROR"
    else:
        lower, upper = string.split("-")
        try:
            lower = int(lower)
            upper = int(upper)
        except:
            return "ERROR"

        return list(range(lower, upper + 1))


def given_map(number):
    if number == 0:
        return "0"
    elif number < 10:
        return "1"
    elif number < 50:
        return "2"
    elif number < 200:
        return "3"
    elif number < 1000:
        return "4"
    return "5"


text_label = Label(text="Enter the year or range, eg. (2018 or 2018-2023)")
regression_label = Label(text="Enter a year and input the C4K Database as csv. Output will be two csv files")
graph_label = Label(text="Enter a year or range and input the C4K Database as csv. Output will be a png")
zipcode_label = Label(
    text="Enter a year and input a csv with header titled Zipcode and various 5-digit Zipcodes in the columns. Output will be a csv file")
inputtxt = Text(height=1, width=20)
spacer = Label(text="")
spacer2 = Label(text="")

btn_1 = Button(m, text="Zipcode Demand Classifier", command=classifier_handler)
btn_2 = Button(m, text="Graph Generator", command=map_handler)
btn_3 = Button(m, text="Demand Prediction", command=regression_handler)

spacer.pack(pady=10)
text_label.pack(pady=5)
inputtxt.pack(pady=5)
spacer2.pack(pady=10)
regression_label.pack()
btn_3.pack(pady=10)
graph_label.pack()
btn_2.pack(pady=10)
zipcode_label.pack()
btn_1.pack(pady=10)
m.mainloop()
