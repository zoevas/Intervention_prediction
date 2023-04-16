# This is a demo where the user gives new input for the vitals and demographics for a patient
# and this input is given to a trained model saved as model.h5 for making a prediction if
# the patient will need mechanical ventilation (0:   "The onset of intervention",  1:'The patient is under control',
#   2:  "Staying on intervention",   3: "Staying off intervention") in a specific hour.
import statistics

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gradio as gr
import numpy as np
import pandas as pd
import statistics as st
from keras.models import load_model
import json
import torch

import tensorflow as tf

# Dictionary for final prediction result
outputdict = {0:   "The onset of intervention",
  1:'The patient is under control',
  2:  "Staying on intervention",
  3: "Staying off intervention"}

GAP_TIME = 6

# HASHMAPS for glascow coma score one hot encodings
#https://www.scaler.com/topics/hashmap-in-python/
gcs_eyeopening = {
    1: [1,  0, 0 , 0],
    2: [0, 1, 0, 0 ],
    3: [0, 0, 1, 0],
    4: [0, 0 , 0, 1]
}
gcs_motor_response = {
    1:[1, 0, 0, 0, 0, 0],
    2:[0, 1, 0, 0, 0, 0],
    3:[0,  0, 1, 0, 0, 0],
    4:[0, 0, 0, 1, 0, 0],
    5:[0, 0, 0, 0, 1, 0],
    6:[0, 0, 0, 0, 0, 1]
}

gcs_coma_scale_total = {
    3:  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    4:  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    5:  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    6:  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    7:  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    8:  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    9:  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    10: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    11: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    12: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

gcs_coma_scale_verbal = {
    1: [1, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0],
    3: [0, 0, 1, 0, 0],
    4: [0, 0, 0, 1, 0],
    5: [0, 0, 0, 0, 1]
}

# Map variables to the same metric:
UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]


def repeat_item(element, count):
    myList = []
    for i in range(count):
         myList.append(element)
    return myList

def convert_function(elementName, value, valueArr):
    if (elementName == 'Weight'):
        value = value * 0.45359237
        value= minmax(value, valueArr)
    else:
        value= minmax(value, valueArr)
    return value

def extend_item(elementName, value , valueCounter, vitalsArrcounter, vitalsArr):
    vitalsArr[vitalsArrcounter] = 1.0
    convValue = convert_function(elementName, value[valueCounter-1], value)
    #print('convValue=', convValue)
    vitalsArr[vitalsArrcounter + 1] = convValue
    vitalsArr[vitalsArrcounter + 2] = -0.21233978  # time since the last measurement. I added 0 such as measured now
        #np.std(value, dtype=np.float64) #std_time_since_measurement(value, value[valueCounter-1])#statistics.stdev(value)

def extend_gcs_item(elementName, valueArr ,counter, vitalsArr):
    npArr = np.array(valueArr)
    for i in npArr:
        vitalsArr[counter] = i
        counter +=1

def minmax(val, x):
    mins = x.min()
    maxes = x.max()
    if maxes == mins:
        return val
    x_std = (val - mins) / (maxes - mins)
    return x_std

def std_time_since_measurement(val, x):
    idx = pd.IndexSlice
    x = np.where(x==100, 0, x)
    means = x.mean()
    stds = x.std() + 0.0001
    x_std = (val - means)/stds
    print('x_std=', x_std)
    return x_std

#DBP, FIO, Glucose, Heart Rate, Mean_Blood_Pressure, OXygen_Saturation, Respiratory rate, Systolic_Blood_Pressure, Temperature, Weight, pH,......, , ethnicity
#sample_input = np.array([78.0, 0.69, 162.0, 92.0, 75.0, 92.0, 11.0, 70.0, 37.4, 110.0, 7.41, 4.0, 6.0 ,13.0, 1.0, 'M' , 'ASIAN'])

# Tailor the new input, namely the vitals for 6 hours, to the features expected by the trained model.
# For the time series, the user gives the values for the first 6 hours.
def feature_process(Diastolic_Blood_Pressure1,Diastolic_Blood_Pressure2,
                    Diastolic_Blood_Pressure3, Diastolic_Blood_Pressure4,
                    Diastolic_Blood_Pressure5, Diastolic_Blood_Pressure6,
                    Fraction_Inspired_Oxygen1, Fraction_Inspired_Oxygen2,
                    Fraction_Inspired_Oxygen3, Fraction_Inspired_Oxygen4,
                    Fraction_Inspired_Oxygen5, Fraction_Inspired_Oxygen6,
                    Glucose1, Glucose2,
                    Glucose3, Glucose4,
                    Glucose5, Glucose6,
                    Heart_Rate1, Heart_Rate2,
                    Heart_Rate3, Heart_Rate4,
                    Heart_Rate5, Heart_Rate6,
                    Mean_Blood_Pressure1, Mean_Blood_Pressure2,
                    Mean_Blood_Pressure3, Mean_Blood_Pressure4,
                    Mean_Blood_Pressure5, Mean_Blood_Pressure6,
                    Oxygen_Saturation1, Oxygen_Saturation2,
                    Oxygen_Saturation3, Oxygen_Saturation4,
                    Oxygen_Saturation5, Oxygen_Saturation6,
                    Respiratory_rate1, Respiratory_rate2,
                    Respiratory_rate3, Respiratory_rate4,
                    Respiratory_rate5, Respiratory_rate6,
                    Systolic_Blood_Pressure1, Systolic_Blood_Pressure2,
                    Systolic_Blood_Pressure3, Systolic_Blood_Pressure4,
                    Systolic_Blood_Pressure5, Systolic_Blood_Pressure6,
                    Temperature1, Temperature2,
                    Temperature3, Temperature4,
                    Temperature5, Temperature6,
                    Weight1, Weight2, Weight3, Weight4, Weight5, Weight6,
                    pH1, pH2, pH3, pH4, pH5, pH6,  Glascow_eye_opening, Glascow_motor_response, Glascow_total,
                    Glasgow_verbal_response, gender):

    arr = np.zeros((1, 6, 83))
    Diastolic_Blood_Pressure = np.array([Diastolic_Blood_Pressure1, Diastolic_Blood_Pressure2,
                                         Diastolic_Blood_Pressure3, Diastolic_Blood_Pressure4,
                                         Diastolic_Blood_Pressure5, Diastolic_Blood_Pressure6])
    Fraction_Inspired_Oxygen = np.array([Fraction_Inspired_Oxygen1, Fraction_Inspired_Oxygen2,
                                         Fraction_Inspired_Oxygen3, Fraction_Inspired_Oxygen4,
                                         Fraction_Inspired_Oxygen5, Fraction_Inspired_Oxygen6])
    Glucose = np.array([Glucose1, Glucose2,
                        Glucose3, Glucose4,
                        Glucose5, Glucose6])
    Heart_Rate = np.array([Heart_Rate1, Heart_Rate2,
                        Heart_Rate3, Heart_Rate4,
                        Heart_Rate5, Heart_Rate6])
    Mean_Blood_Pressure = np.array([Mean_Blood_Pressure1, Mean_Blood_Pressure2,
                        Mean_Blood_Pressure3, Mean_Blood_Pressure4,
                        Mean_Blood_Pressure5, Mean_Blood_Pressure6])
    Oxygen_Saturation = np.array([Oxygen_Saturation1, Oxygen_Saturation2,
                        Oxygen_Saturation3, Oxygen_Saturation4,
                        Oxygen_Saturation5, Oxygen_Saturation6])
    Respiratory_rate = np.array([Respiratory_rate1, Respiratory_rate2,
                        Respiratory_rate3, Respiratory_rate4,
                        Respiratory_rate5, Respiratory_rate6])
    Systolic_Blood_Pressure = np.array([Systolic_Blood_Pressure1, Systolic_Blood_Pressure2,
                        Systolic_Blood_Pressure3, Systolic_Blood_Pressure4,
                        Systolic_Blood_Pressure5, Systolic_Blood_Pressure6])
    Temperature = np.array([Temperature1, Temperature2,
                        Temperature3, Temperature4,
                        Temperature5, Temperature6])
    Weight = np.array([Weight1, Weight2,
                        Weight3, Weight4,
                        Weight5, Weight6])
    pH = np.array([pH1, pH2,
                    pH3, pH4,
                    pH5, pH6])

    vitalsArr_holder = {}

    for k in range(1, GAP_TIME + 1):
        vitalsArr_holder['my_var_' + str(k)] = np.zeros(83)
        extend_item(elementName='Diastolic_Blood_Pressure', value=Diastolic_Blood_Pressure, valueCounter=k, vitalsArrcounter=0, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Fraction_Inspired_Oxygen', value=Fraction_Inspired_Oxygen, valueCounter=k, vitalsArrcounter=3, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Glucose', value=Glucose, valueCounter=k, vitalsArrcounter=6, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Heart Rate', value=Heart_Rate, valueCounter=k, vitalsArrcounter=9, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        #Height
        vitalsArr_holder['my_var_' + str(k)][12] = 0.0
        vitalsArr_holder['my_var_' + str(k)][13] = 0.704
        vitalsArr_holder['my_var_' + str(k)][14] = -0.9982
        extend_item(elementName='Mean_Blood_Pressure', value=Mean_Blood_Pressure, valueCounter=k, vitalsArrcounter=15, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Oxygen_Saturation', value=Oxygen_Saturation, valueCounter=k, vitalsArrcounter=18, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Respiratory_rate', value=Respiratory_rate, valueCounter=k, vitalsArrcounter=21, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Systolic_Blood_Pressure', value=Systolic_Blood_Pressure, valueCounter=k, vitalsArrcounter=24, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Temperature', value=Temperature, valueCounter=k, vitalsArrcounter=27, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='Weight', value=Weight, valueCounter=k, vitalsArrcounter=30, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_item(elementName='pH', value=pH, valueCounter=k, vitalsArrcounter=33, vitalsArr=vitalsArr_holder['my_var_' + str(k)])

        print('gcs_eyeopening', gcs_eyeopening.get(round(Glascow_eye_opening)))
        print('gcs_motor_response', gcs_motor_response.get(round(Glascow_motor_response)))
        print('gcs_coma_scale_total', round(Glascow_total))
        print('gcs_coma_scale_total', gcs_coma_scale_total.get(round(Glascow_total)))
        print('gcs_coma_scale_verbal', gcs_coma_scale_verbal.get(round(Glasgow_verbal_response)))
        gcs_data = np.array(
            [Glascow_eye_opening,
            Glascow_motor_response,
            Glascow_total,
            Glasgow_verbal_response])
        gcsdataset = pd.DataFrame({'subject_id': 1, 'Glascow coma scale eye opening': gcs_data[0], 'Glascow coma scale motor response': gcs_data[1],'Glascow coma scale total': gcs_data[2],'Glascow coma scale verbal response': gcs_data[3]},
                              index=['Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response'])


        extend_gcs_item(elementName='Glascow_eye_opening', valueArr=gcs_eyeopening.get(round(Glascow_eye_opening)), counter=36, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_gcs_item(elementName='Glascow_motor_response', valueArr=gcs_motor_response.get(round(Glascow_motor_response)), counter=40, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_gcs_item(elementName='Glascow_total', valueArr=gcs_coma_scale_total.get(round(Glascow_total)), counter=46, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        extend_gcs_item(elementName='Glascow_verbal_response', valueArr=gcs_coma_scale_verbal.get(round(Glasgow_verbal_response)), counter=59, vitalsArr=vitalsArr_holder['my_var_' + str(k)])
        #calculate_ethnicity()
        #calculate_gender()
        #  vitalsRow[45] ='2101-10-20 19:10:11'.astype('datetime64').apply(lambda x : x.hour)
        #intime
        vitalsArr_holder['my_var_' + str(k)][64] = 0
        if gender == 'M':
            vitalsArr_holder['my_var_' + str(k)][65] = 0
            vitalsArr_holder['my_var_' + str(k)][66] = 1
        else:
            vitalsArr_holder['my_var_' + str(k)][65] = 1
            vitalsArr_holder['my_var_' + str(k)][66] = 0

        #Some hardcoded values are given for the sake of the demo, this input should be given dynamically by the UI (pending)
        #admission age
        vitalsArr_holder['my_var_' + str(k)][67] = 0
        vitalsArr_holder['my_var_' + str(k)][68] = 1
        vitalsArr_holder['my_var_' + str(k)][69] = 0
        vitalsArr_holder['my_var_' + str(k)][70] = 0

        #ethnicity
        vitalsArr_holder['my_var_' + str(k)][71] = 0
        vitalsArr_holder['my_var_' + str(k)][72] = 1
        vitalsArr_holder['my_var_' + str(k)][73] = 0
        vitalsArr_holder['my_var_' + str(k)][74] = 0
        vitalsArr_holder['my_var_' + str(k)][75] = 0
        vitalsArr_holder['my_var_' + str(k)][76] = 0

        #first_careunit
        vitalsArr_holder['my_var_' + str(k)][77] = 1
        vitalsArr_holder['my_var_' + str(k)][78] = 0
        vitalsArr_holder['my_var_' + str(k)][79] = 0
        vitalsArr_holder['my_var_' + str(k)][80] = 0
        vitalsArr_holder['my_var_' + str(k)][81] = 0

        #intervention
        vitalsArr_holder['my_var_' + str(k)][82] = 0

    model = load_model('/app/src/mymodel.h5')
    #print(model.summmary())

    input = np.array([[vitalsArr_holder['my_var_1'],vitalsArr_holder['my_var_2'],
                       vitalsArr_holder['my_var_3'], vitalsArr_holder['my_var_4'],
                       vitalsArr_holder['my_var_5'], vitalsArr_holder['my_var_6']]])
    print('input:', input)
    #predict_results = model.predict(input)
    predict_results = model.predict(input)
    print('predict results=', predict_results)
    #argmax is used for finding the value with the largest probability
    print('argmax:', np.argmax(predict_results))

    final_result = np.argmax(predict_results, axis=-1)
    print('vitalsArr_holder["my_var_1"]', vitalsArr_holder['my_var_1'])
    print('final_result', final_result)
    #return "predict_results: " + predict_results
    if (final_result == 0):
        return "The onset of intervention"
    elif (final_result == 1):
        return  "Patient under control"
    elif (final_result == 2):
        return  "Staying on intervention"
    elif (final_result == 3):
        return  "Staying off intervention"

   # return outputdict[final_result]

def to_scientific(x):
    return np.format_float_scientific(x, precision = 4, exp_digits=3)

#Using the gradio io, a UI is launched in localhost:8080 where the user can give input values to the ML model for making the prediction
if __name__ == '__main__':
    model = load_model('/app/src/mymodel.h5')
    #print(model.summmary())

    test_input = np.random.random((1, 6, 83))
    #print('x_window=', x_window)
   # predict_results = model.predict(test_input)
   # print('predict_results', predict_results)

    scientific = np.vectorize(to_scientific)
    #print('x_window after', x_window)
   # predict_proba_results = model.
   # print('predict_proba_results', predict_proba_results)
    iface = gr.Interface(fn=feature_process, inputs=[# DBP
                                                     gr.inputs.Slider(0.00, 375.00, step=0.1), gr.inputs.Slider(0.00, 375.00,  step=0.1),
                                                     gr.inputs.Slider(0.00, 375.00,  step=0.1), gr.inputs.Slider(0.00, 375.00,  step=0.1),
                                                     gr.inputs.Slider(0.0, 375.00,  step=0.1), gr.inputs.Slider(0.00, 375.00,  step=0.1),
                                                     gr.inputs.Slider(0, 1),gr.inputs.Slider(0, 1),
                                                     gr.inputs.Slider(0, 1), gr.inputs.Slider(0, 1),
                                                     gr.inputs.Slider(0, 1), gr.inputs.Slider(0, 1),
                                                     gr.inputs.Slider(33, 2000),gr.inputs.Slider(33, 2000),
                                                     gr.inputs.Slider(33, 2000),gr.inputs.Slider(33, 2000),
                                                     gr.inputs.Slider(33, 2000),gr.inputs.Slider(33, 2000),
                                                     gr.inputs.Slider(0, 350), gr.inputs.Slider(0, 350),
                                                     gr.inputs.Slider(0, 350), gr.inputs.Slider(0, 350),
                                                     gr.inputs.Slider(0, 350), gr.inputs.Slider(0, 350),
                                                     gr.inputs.Slider(14, 330, step=0.00000000000001),gr.inputs.Slider(14, 330, step=0.00000000000001),
                                                     gr.inputs.Slider(14, 330, step=0.00000000000001),gr.inputs.Slider(14, 330, step=0.00000000000001),
                                                     gr.inputs.Slider(14, 330, step=0.00000000000001),gr.inputs.Slider(14, 330, step=0.00000000000001),
                                                     gr.inputs.Slider(0, 100), gr.inputs.Slider(0, 100),
                                                     gr.inputs.Slider(0, 100), gr.inputs.Slider(0, 100),
                                                     gr.inputs.Slider(0, 100), gr.inputs.Slider(0, 100),
                                                     gr.inputs.Slider(0, 300), gr.inputs.Slider(0, 300),
                                                     gr.inputs.Slider(0, 300), gr.inputs.Slider(0, 300),
                                                     gr.inputs.Slider(0, 300), gr.inputs.Slider(0, 300),
                                                     gr.inputs.Slider(0, 375, step=0.01), gr.inputs.Slider(0, 375, step=0.01),
                                                     gr.inputs.Slider(0, 375, step=0.01), gr.inputs.Slider(0, 375, step=0.01),
                                                     gr.inputs.Slider(0, 375, step=0.01), gr.inputs.Slider(0, 375, step=0.01),
                                                     gr.inputs.Slider(26, 45,step=0.000000000000001), gr.inputs.Slider(26, 45,step=0.000000000000001),
                                                     gr.inputs.Slider(26, 45,step=0.000000000000001), gr.inputs.Slider(26, 45,step=0.000000000000001),
                                                     gr.inputs.Slider(26, 45,step=0.000000000000001), gr.inputs.Slider(26, 45,step=0.000000000000001),
                                                     gr.inputs.Slider(20, 300,step=0.001), gr.inputs.Slider(20, 300,step=0.001),
                                                     gr.inputs.Slider(20, 300,step=0.001), gr.inputs.Slider(20, 300,step=0.001),
                                                     gr.inputs.Slider(20, 300,step=0.001), gr.inputs.Slider(20, 300,step=0.001),
                                                     gr.inputs.Slider(6.3, 8.4),  gr.inputs.Slider(6.3, 8.4),
                                                     gr.inputs.Slider(6.3, 8.4),  gr.inputs.Slider(6.3, 8.4),
                                                     gr.inputs.Slider(6.3, 8.4),  gr.inputs.Slider(6.3, 8.4),
                                                     gr.inputs.Slider(1, 4),  gr.inputs.Slider(1, 6), gr.inputs.Slider(1, 15),
                                                     gr.inputs.Slider(1, 5),
                                                     gr.Dropdown(['M', 'F'])],  outputs="text",  cache_examples = True)
    #  gr.Radio(["Asian", "Black", "Other", "White"
    iface.launch(server_name="0.0.0.0", server_port=8080)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
