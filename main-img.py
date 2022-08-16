import cv2
import numpy as np

max_value = 255
max_value_H = 180
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_detection_name = 'trackbars'
low_H_name = 'L-H'
low_S_name = 'L-S'
low_V_name = 'L-V'
high_H_name = 'U-H'
high_S_name = 'U-S'
high_V_name = 'U-V'

# essas funções são usadas para evitar que, por exemplo, o valor de low hue seja maior do que o valor do high hue

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

# cria uma janela com os parâmetros a serem alterados
cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar) # lower hue
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar) # lower saturation
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar) # lower value
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar) # upper hue
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar) # upper saturation
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar) # upper value

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    frame = cv2.imread('cubo-vermelho.jpg')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)) # define a máscara usando a img em hsv
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel) # usa um quadrado kernel de 5x5 pra eliminar um pouco de noise da imagem

    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # a variavel "_" é a variável das hierarquias dos contornos, mas ela n foi usada, por isso é só um "_"

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) # os contornos ficavam com linhas meio tortas, por isso uso a aproximação pra deixar elas retas
        x = approx.ravel()[0] # pega o x do primeiro ponto do contorno
        y = approx.ravel()[1] # pega o y do primeiro ponto do contorno
        if area > 400: # usa uma area > 400 como outra forma de eliminar noise da img
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5) # desenha os contornos aproximados

            cv2.putText(frame, f'Points: {len(approx)}', (x, y-50), font, 1, (0, 0, 0))
            cv2.putText(frame, f'Area: {area}', (x, y-10), font, 1, (0, 0, 0))


    cv2.imshow('frame', frame)

    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

