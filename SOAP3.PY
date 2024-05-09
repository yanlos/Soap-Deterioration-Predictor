# Thought process:
# The reason why I did it this way was because I've always wanted to try out computer vision libraries and this seemed 
# like a perfect opportunity to use it in this simulation. I thought it would be interesting to try use CV to track 
# the deterioration of a bar of soap (in my case a bath bomb that would melt much faster than soap)


# WHAT I CAN IMPROVE IN THE FUTURE 
# 1. CONFIGURE HOV VALUES MORE ACCURATELY SUCH AS UPPER WHITE SO THAT BACKGROUND WATER DOESN'T GET PICKED UP 
# 2. USE SOAP THATS A DIFFERENT COLOR THAN THE WATER
# 3. STABILIZE THE CAMERA 

# to use "pip3 install" matplotlib AND cv2 AND numpy AND scikit-learn

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# we want to go from RGB to HSV (hue, saturation, and value) in order to 
# color object tracking in https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html


def prediction(frame_numbers, black_areas, frame_count):
    # reshape to use sklearn after turning into arrays https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # https://stackoverflow.com/questions/47761744/cant-do-linear-regression-in-scikit-learn-due-to-reshaping-issue
    Xaxis = np.array(frame_numbers).reshape(-1, 1)
    Yaxis = np.array(black_areas).reshape(-1, 1)
    
    # wrapper for ninput validation
    # using numpy for random # generator using 70/30 
    X_train, X_test, y_train, y_test = train_test_split(Xaxis, Yaxis, test_size=0.3, random_state=np.random.RandomState())
    
    # we're going to use a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # our "prediction" after training the model 
    future_frames = np.array(range(frame_count, frame_count+30)).reshape(-1, 1)
    predicted_areas = model.predict(future_frames)
    
    # add existing and future data to use it for plotting 
    all_frames = np.concatenate((Xaxis, future_frames))
    all_areas = np.concatenate((Yaxis, predicted_areas))
    
    return all_frames, all_areas


"""
We want to go from RGB to HSV (hue, saturation, and value) in order to 
See color object tracking in https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

returns res (result) the frame with just the soap   
"""
def get_soap(frame, lower_white, upper_white):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # only get white colors (soap color), can be done for any color like blue in the example
    # use eye dropper/color picker for values
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res

"""
It's called black area (see photo black_soap.png) because that's how the original attempt I wrote up
was displayed. This is the same, just in process video rather than showing us the black area. 
We display the green edge detector when we do the contour, rather than the black to visualize it better.
CV2 does the turning the video into black OR white and that's how we can find our soap area.

returns black_area: how much of the video is covered by black
        black_mask: the map of where these black covered parts are
        these are both processed by cv2 
"""
def calculate_black_area(res, lower_black=np.array([0, 0, 0]), upper_black=np.array([180, 255, 50])):
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_area = cv2.countNonZero(black_mask)
    return black_area, black_mask


# def plot_graph(frame_numbers, black_areas):
#     plt.figure(figsize=(10, 6))
#     plt.plot(frame_numbers, black_areas, label='DEGRADATION over TIME', color='red')

# Plotting the black area (px) over time (frames)
def plot_graph(all_frames, all_areas, original_frame_numbers, original_black_areas):
    plt.figure(figsize=(12, 6))

    # normal plot (colored over blue)
    plt.plot(original_frame_numbers, original_black_areas, label='Actual', color='green', linewidth=2)
    
    # enlarge the x axis
    # plt.xlim(0, max(all_frames) * 1.1)
    # this is going to be our preducted line following 
    plt.plot(all_frames, all_areas, '--', label='Predicted', color='red', linewidth=2, markersize=5)
    
    plt.xlabel('TIME (frames)')
    plt.ylabel('SOAP AREA (pixels)')
    plt.legend()
    plt.show()


def process_video(video_path, lower_white, upper_white, box_area):
    cap = cv2.VideoCapture(video_path)
    black_areas = []
    frame_numbers = []    
    frame_count = 0

    if not cap.isOpened():
        print(".MP4 NOT FOUND")
        exit()

    # start
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = box_area
        box_frame = frame[y:y+h, x:x+w]

        # remove_water(box_frame)
        res = get_soap(box_frame, lower_white, upper_white)
        black_area, black_mask = calculate_black_area(res)

        # Find contours of the black areas, RETR_EXTERNAL is an attribute from CV2 where we get the furthest edges
        # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            adjusted_contour = contour + np.array([x, y])
            cv2.drawContours(frame, [adjusted_contour], -1, (0, 255, 0), 1) 

        # real time counter of pixels
        cv2.putText(frame, f"SOAP AREA: {black_area} pixels", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # visulization of the red frame that we are counting the area in 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  
        cv2.imshow('SOAP_SIMULATION_YAN_LOS', frame)

        black_areas.append(black_area)
        frame_numbers.append(frame_count)
        frame_count += 1 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    all_frames, all_areas = prediction(frame_numbers, black_areas, frame_count*2)
    # plot_graph(frame_numbers, black_areas)

    # Plot the results including predictions
    plot_graph(all_frames, all_areas, frame_numbers, black_areas)

if __name__ == "__main__":
    # ajdustable parameters: only one that matters for white is the third upper_white one, can toggle it between 105-120
    lower_white = np.array([0, 0, 50])  
    upper_white = np.array([180, 255, 115])  
    box_area = (1200, 1100, 775, 775) 
    
    video_path = '/Users/yanlos/Desktop/soaptub/soaptub.mp4'
    process_video(video_path, lower_white, upper_white, box_area)