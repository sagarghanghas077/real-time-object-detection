import cv2

def enhance_low_light(frame):
    """
    Improves visibility in low-light conditions using histogram equalization
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.equalizeHist(v)

    hsv_enhanced = cv2.merge((h, s, v))
    enhanced_frame = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced_frame
