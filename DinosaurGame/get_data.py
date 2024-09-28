from selenium import webdriver
import time
import numpy as np
import cv2
from mss import mss

#Open the Chrome Browser(default)
driver = webdriver.Chrome()

# Sample game from Web
driver.get("https://chromedino.com/")

driver.find_element("tag name", "body").send_keys(" ")

time.sleep(4)

dt = 0.016
def getScreen():
    bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}
    sct = mss()
    raw_screen = np.array(sct.grab(bounding_box))
    gray_screen = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2GRAY)
    return gray_screen

example_screen = getScreen()
screen_shape = example_screen.shape



actions_arr = None
reward_arr = None

try:
    screen_arr = np.load("screen_data.npy")
    print(screen_arr.shape)
    print(screen_arr[0])
except:
    screen_arr = np.empty((0, *screen_shape))

def play(for_t=10):
    global screen_arr
    t = 0
    while t < for_t:
        screen =    getScreen()

        screen_arr = np.vstack((screen_arr, screen[np.newaxis, ...]))
        time.sleep(dt)

        t += (1 * dt)
    np.save("screen_data.npy",screen_arr)

play(10)
driver.quit()
