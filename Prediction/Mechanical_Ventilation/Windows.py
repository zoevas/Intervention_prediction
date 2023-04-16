import numpy as np
SLICE_SIZE = 6
PREDICTION_WINDOW = 4
MAX_LEN = 240
OUTCOME_TYPE = 'all'
NUM_CLASSES = 4
GAP_TIME = 6

CHUNK_KEY = {'ONSET': 0, 'CONTROL': 1, 'ON_INTERVENTION': 2, 'WEAN': 3}


def make_3d_tensor_slices(X_tensor, Y_tensor, lengths):

    num_patients = X_tensor.shape[0]
    print ('num_patients = ', num_patients)
    timesteps = X_tensor.shape[1]
    num_features = X_tensor.shape[2]
    X_tensor_new = np.zeros((lengths.sum(), SLICE_SIZE, num_features + 1))
    Y_tensor_new = np.zeros((lengths.sum()))

    current_row = 0

    for patient_index in range(num_patients):
        x_patient = X_tensor[patient_index]
        y_patient = Y_tensor[patient_index]
        length = lengths[patient_index]
       # # print('x_patient.shape', x_patient.shape)
        range2 = range(length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE)
      #   print('length = ', length)
       #  print('range(length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE)', length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE)

        for timestep in range(length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE):
           # print('timestep', timestep)
            x_window = x_patient[timestep:timestep+SLICE_SIZE]
            y_window = y_patient[timestep:timestep+SLICE_SIZE]
          #  print('x_window.shape', x_window.shape)
           # print('x_window.ndim', x_window.ndim)
           # print('y_window.shape', y_window.shape)
           # print('y_window.ndim', y_window.ndim)
          #  print('before x_window', x_window)
            x_window = np.column_stack((x_window, y_window))
           # print('x_window column_stack shape', x_window.shape)
           # print('x_window', np.column_stack((x_window, y_window)))
           # print('y_window', y_window)
            result_window = y_patient[timestep+SLICE_SIZE+GAP_TIME:timestep+SLICE_SIZE+GAP_TIME+PREDICTION_WINDOW]
          #  print('result_window', result_window.reshape(-1))
          #  print('np.diff(result_window)', np.diff(result_window.reshape(-1)))
          #  print('x_window', x_window)
            result_window_diff = set(np.diff(np.diff(result_window.reshape(-1))))
            #if 1 in result_window_diff: pdb.set_trace()
            gap_window = y_patient[timestep+SLICE_SIZE:timestep+SLICE_SIZE+GAP_TIME]
            gap_window_diff = set(np.diff(gap_window.reshape(-1)))

            #print result_window, result_window_diff

            if OUTCOME_TYPE == 'binary':
                if max(gap_window) == 1:
                    result = None
                elif max(result_window) == 1:
                    result = 1
                elif max(result_window) == 0:
                    result = 0
                if result != None:
                    X_tensor_new[current_row] = x_window
                    Y_tensor_new[current_row] = result
                    current_row += 1

            else:
                if 1 in gap_window_diff or -1 in gap_window_diff:
                    result = None
                elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 0):
                    result = CHUNK_KEY['CONTROL']
                elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 1):
                    result = CHUNK_KEY['ON_INTERVENTION'] # staying on intervention
                elif 1 in result_window_diff:
                    result = CHUNK_KEY['ONSET']
                elif -1 in result_window_diff:
                    result = CHUNK_KEY['WEAN'] #  the mechanical ventilation will be removed in this window
                else:
                    result = None

                if result != None:
                    X_tensor_new[current_row] = x_window
                    Y_tensor_new[current_row] = result
                    current_row += 1

    X_tensor_new = X_tensor_new[:current_row,:,:]
    Y_tensor_new = Y_tensor_new[:current_row]

    print('X_tensor_new.shape', X_tensor_new.shape)

    return X_tensor_new, Y_tensor_new
