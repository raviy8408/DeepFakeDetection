def increase_brightness(img, value=30):
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def adjust_brightness_and_contrast(img, alpha=1.3, beta=40, gamma=0.8):
    import cv2
    import numpy as np

    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    final_image = cv2.LUT(new_image, lookUpTable)

    return final_image
